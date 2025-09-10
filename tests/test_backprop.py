import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from main import Reporter
from network.entities import Neuron, Synapse
from network.graph import Graph
from network.backprop import Backpropagator


class TestBackpropagator(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.zero = torch.tensor(0.0)
        Reporter._metrics = {}

    def _build_graph(self):
        n1 = Neuron(zero=self.zero)
        n2 = Neuron(zero=self.zero)
        n1.update_gate(torch.tensor(0.5))
        n2.update_gate(torch.tensor(0.6))
        w1 = torch.tensor(1.0, requires_grad=True)
        w2 = torch.sqrt(torch.tensor(2.0, requires_grad=True))
        n1.update_weight(w1, reporter=Reporter)
        n2.update_weight(w2, reporter=Reporter)
        n1.record_local_loss(n1.weight ** 2)
        n2.record_local_loss(n2.weight ** 2)
        n1.update_latency(torch.tensor(0.1))
        n2.update_latency(torch.tensor(0.2))
        s1 = Synapse(zero=self.zero)
        s2 = Synapse(zero=self.zero)
        s1.gate = torch.tensor(0.7)
        s2.gate = torch.tensor(0.8)
        w3 = torch.sqrt(torch.tensor(3.0, requires_grad=True))
        w4 = torch.sqrt(torch.tensor(4.0, requires_grad=True))
        s1.update_weight(w3, reporter=Reporter)
        s2.update_weight(w4, reporter=Reporter)
        s1.update_cost(s1.weight ** 2)
        s2.update_cost(s2.weight ** 2)
        s1.update_latency(torch.tensor(0.3))
        s2.update_latency(torch.tensor(0.4))
        g = Graph(reporter=Reporter)
        g.add_neuron("n1", n1)
        g.add_neuron("n2", n2)
        g.add_synapse("s1", "n1", "n2", s1)
        g.add_synapse("s2", "n2", "n1", s2)
        path = [n1, s1, n2]
        return g, path, n1, n2, s1, s2

    def test_hard_vs_soft_masks(self):
        g, path, n1, n2, s1, s2 = self._build_graph()
        bp = Backpropagator(reporter=Reporter)

        gates = bp.build_active_subgraph(g, path, "hard")
        self.assertEqual(Reporter.report("active_edges"), 1)
        self.assertEqual(Reporter.report("active_vertices"), 2)
        loss = bp.compute_sample_loss(g, gates)
        expected = (
            n1.gate * n1.last_local_loss
            + n2.gate * n2.last_local_loss
            + s1.gate * s1.c_e
        )
        print("Hard routing loss:", loss)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.allclose(Reporter.report("sample_loss"), expected))

        gates = bp.build_active_subgraph(g, path, "soft")
        self.assertEqual(Reporter.report("active_edges"), 2)
        self.assertEqual(Reporter.report("active_vertices"), 2)
        loss = bp.compute_sample_loss(g, gates)
        expected = expected + s2.gate * s2.c_e
        print("Soft routing loss:", loss)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.allclose(Reporter.report("sample_loss"), expected))

    def test_gradient_consistency(self):
        g, path, n1, n2, s1, s2 = self._build_graph()
        bp = Backpropagator(reporter=Reporter)
        gates = bp.build_active_subgraph(g, path, "soft")
        loss = bp.compute_sample_loss(g, gates)
        grads = bp.compute_gradients(g, gates, loss)

        total_cost = loss
        for nid, neuron in g.neurons.items():
            gate = gates["g_v"][nid]
            lam = neuron.lambda_v
            w = neuron.weight
            total_cost = total_cost + gate * lam * (torch.abs(w) + 0.5 * w.pow(2))
        for sid, (_, _, synapse) in g.synapses.items():
            gate = gates["g_e"][sid]
            lam = synapse.lambda_e
            w = synapse.weight
            total_cost = total_cost + gate * lam * 0.5 * w.pow(2)

        expected_grads = torch.autograd.grad(
            total_cost,
            [n1.weight, n2.weight, s1.weight, s2.weight],
            allow_unused=True,
        )
        print("Computed gradients:", grads)
        print("Expected gradients:", expected_grads)
        self.assertTrue(torch.allclose(grads["neurons"]["n1"], expected_grads[0]))
        self.assertTrue(torch.allclose(grads["neurons"]["n2"], expected_grads[1]))
        self.assertTrue(torch.allclose(grads["synapses"]["s1"], expected_grads[2]))
        self.assertTrue(torch.allclose(grads["synapses"]["s2"], expected_grads[3]))
        self.assertIsNotNone(
            Reporter.report(f"neuron_{id(n1)}_grad_norm")
        )
        self.assertIsNotNone(
            Reporter.report(f"synapse_{id(s1)}_grad_norm")
        )


if __name__ == "__main__":
    unittest.main()
