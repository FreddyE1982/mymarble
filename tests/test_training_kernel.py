import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from main import Reporter
from network.entities import Neuron, Synapse
from network.graph import Graph
from network.backprop import Backpropagator
from network.telemetry import TelemetryUpdater
from network.training import TrainingKernel


class TestTrainingKernel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        Reporter._metrics = {}
        self.zero = torch.tensor(0.0)

    def _build_graph(self):
        n1 = Neuron(zero=self.zero)
        n2 = Neuron(zero=self.zero)
        s = Synapse(zero=self.zero)
        n1.update_gate(torch.tensor(1.0))
        n2.update_gate(torch.tensor(1.0))
        s.gate = torch.tensor(1.0)
        n1.phi_v = torch.tensor(1.0)
        n2.phi_v = torch.tensor(-10.0)
        w1 = torch.tensor(1.0, requires_grad=True)
        w2 = torch.tensor(1.5, requires_grad=True)
        w3 = torch.tensor(2.0, requires_grad=True)
        n1.update_weight(w1, reporter=Reporter)
        n2.update_weight(w2, reporter=Reporter)
        s.update_weight(w3, reporter=Reporter)
        n1.record_local_loss(n1.weight ** 2)
        n2.record_local_loss(n2.weight ** 2)
        s.update_cost(s.weight ** 2)
        g = Graph(reporter=Reporter)
        g.add_neuron("n1", n1)
        g.add_neuron("n2", n2)
        g.add_synapse("s", "n1", "n2", s)
        return g, n1, n2, s

    def test_iteration_updates_weights_and_metrics(self):
        g, n1, n2, s = self._build_graph()
        bp = Backpropagator(reporter=Reporter)
        telemetry = TelemetryUpdater(reporter=Reporter)
        trainer = TrainingKernel(g, bp, telemetry, reporter=Reporter)
        lr_v = torch.tensor(0.05)
        lr_e = torch.tensor(0.07)
        evolution = {"remove_neuron": ("n2", {"b_vn": torch.tensor(0.0), "g_vn": torch.tensor(0.0), "latency": torch.tensor(0.0), "cost": torch.tensor(0.0)})}
        print("Initial weights:", n1.weight, n2.weight, s.weight)
        result = trainer.run_iteration(lr_v=lr_v, lr_e=lr_e, evolution_instructions=evolution)
        print("Updated weights:", n1.weight, n2.weight, s.weight)
        print("Mutation results:", result.get("mutations"))
        self.assertIn("path", result)
        self.assertIn("loss", result)
        self.assertIn("grads", result)
        self.assertNotEqual(n1.weight.item(), 1.0)
        self.assertNotEqual(n2.weight.item(), 1.5)
        self.assertIsNotNone(Reporter.report(f"neuron_{id(n1)}_cumulative_loss"))
        self.assertEqual(Reporter.report("training_iterations"), 1)

    def test_iteration_uses_sample_for_losses(self):
        g, n1, n2, s = self._build_graph()
        def loss_fn(neuron, x):
            return x
        def cost_fn(syn, x):
            return x
        n1.loss_fn = loss_fn
        n2.loss_fn = loss_fn
        s.cost_fn = cost_fn
        bp = Backpropagator(reporter=Reporter)
        telemetry = TelemetryUpdater(reporter=Reporter)
        trainer = TrainingKernel(g, bp, telemetry, reporter=Reporter)
        sample = torch.tensor(3.0)
        trainer.run_iteration(sample=sample, lr_v=torch.tensor(0.0), lr_e=torch.tensor(0.0))
        print("Sample-based loss n1:", n1.last_local_loss)
        print("Sample-based cost s:", s.c_e)
        self.assertAlmostEqual(n1.last_local_loss.item(), 3.0)
        self.assertAlmostEqual(s.c_e.item(), 3.0)


if __name__ == "__main__":
    unittest.main()
