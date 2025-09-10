import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from main import Reporter
from network.entities import Neuron, Synapse
from network.graph import Graph
from network.backprop import Backpropagator


class TestBackpropWorkflow(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        Reporter._metrics = {}
        self.zero = torch.tensor(0.0)

    def _build_graph(self):
        n1 = Neuron(zero=self.zero)
        n2 = Neuron(zero=self.zero)
        n1.update_gate(torch.tensor(0.5))
        n2.update_gate(torch.tensor(0.6))
        n1.update_weight(torch.tensor(1.0, requires_grad=True), reporter=Reporter)
        n2.update_weight(torch.sqrt(torch.tensor(2.0)).detach().requires_grad_(), reporter=Reporter)
        n1.record_local_loss(n1.weight ** 2)
        n2.record_local_loss(n2.weight ** 2)
        n1.update_latency(torch.tensor(0.1))
        n2.update_latency(torch.tensor(0.2))
        n1.phi_v = torch.tensor(10.0)
        n2.phi_v = torch.tensor(-10.0)

        s = Synapse(zero=self.zero)
        s.gate = torch.tensor(0.7)
        s.update_weight(torch.sqrt(torch.tensor(3.0)).detach().requires_grad_(), reporter=Reporter)
        s.update_cost(s.weight ** 2)
        s.update_latency(torch.tensor(0.3))

        g = Graph(reporter=Reporter)
        g.add_neuron("n1", n1)
        g.add_neuron("n2", n2)
        g.add_synapse("s", "n1", "n2", s)

        forward = g.forward(global_loss_target=torch.tensor(0.0))
        path = forward["path"]
        # Restore synapse state to keep gradient connections intact
        s.update_cost(s.weight ** 2)
        s.update_latency(torch.tensor(0.3))
        return g, path, n1, n2, s

    def test_full_backprop_workflow(self):
        g, path, n1, n2, s = self._build_graph()
        bp = Backpropagator(reporter=Reporter)

        gates = bp.build_active_subgraph(g, path, "hard")
        print("Gates:", gates)
        self.assertTrue(torch.allclose(gates["g_v"]["n1"], n1.gate))
        self.assertTrue(torch.allclose(gates["g_v"]["n2"], n2.gate))
        self.assertTrue(torch.allclose(gates["g_e"]["s"], s.gate))

        loss = bp.compute_sample_loss(g, gates)
        expected_loss = (
            n1.gate * n1.last_local_loss
            + n2.gate * n2.last_local_loss
            + s.gate * s.c_e
        )
        print("Loss:", loss)
        self.assertTrue(torch.allclose(loss, expected_loss))

        active = {"graph": g, "g_v": gates["g_v"], "g_e": gates["g_e"], "loss": loss}
        grads = bp.compute_gradients(active)
        print("Gradients:", grads)
        total_cost = loss
        for neuron in [n1, n2]:
            lam = neuron.lambda_v
            w = neuron.weight
            total_cost = total_cost + lam * (torch.abs(w) + 0.5 * w.pow(2))
        lam_e = s.lambda_e
        w_e = s.weight
        total_cost = total_cost + lam_e * 0.5 * w_e.pow(2)
        expected_grads = torch.autograd.grad(total_cost, [n1.weight, n2.weight, s.weight])
        self.assertTrue(torch.allclose(grads["neurons"]["n1"], expected_grads[0]))
        self.assertTrue(torch.allclose(grads["neurons"]["n2"], expected_grads[1]))
        self.assertTrue(torch.allclose(grads["synapses"]["s"], expected_grads[2]))
        self.assertIsNotNone(Reporter.report("routing_adjustment"))
        self.assertGreater(Reporter.report(f"neuron_{id(n1)}_grad_norm"), 0)
        self.assertGreater(Reporter.report(f"synapse_{id(s)}_grad_norm"), 0)

        lr_v = torch.tensor(0.05)
        lr_e = torch.tensor(0.07)
        old_w_n1 = n1.weight.detach().clone()
        old_w_n2 = n2.weight.detach().clone()
        old_w_s = s.weight.detach().clone()
        active["grads"] = grads
        bp.apply_updates(active, lr_v, lr_e)
        print("Updated weights:", n1.weight, n2.weight, s.weight)
        self.assertTrue(torch.allclose(n1.weight, old_w_n1 - lr_v * grads["neurons"]["n1"]))
        self.assertTrue(torch.allclose(n2.weight, old_w_n2 - lr_v * grads["neurons"]["n2"]))
        self.assertTrue(torch.allclose(s.weight, old_w_s - lr_e * grads["synapses"]["s"]))
        self.assertTrue(torch.all(Reporter.report("lr_v") == lr_v))
        self.assertTrue(torch.all(Reporter.report("lr_e") == lr_e))


if __name__ == "__main__":
    unittest.main()
