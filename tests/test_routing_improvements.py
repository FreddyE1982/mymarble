import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from routing.improvements import GateAdjuster
from network.entities import Neuron, Synapse
from network.graph import Graph


class TestRoutingImprovements(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}
        self.zero = torch.tensor(0.0)

    def test_gate_adjuster_updates_threshold(self):
        neuron = Neuron(zero=self.zero, activation_threshold=torch.tensor(0.5, requires_grad=True))
        adjuster = GateAdjuster(reporter=main.Reporter, learning_rate=0.1)
        stats = {
            "gradient": torch.tensor(1.0),
            "latency": torch.tensor(0.2),
            "cost": torch.tensor(0.3),
        }
        new_thresh = adjuster.adjust_gate(neuron, stats)
        print("Influence metric:", main.Reporter.report("routing_adjustment"))
        self.assertAlmostEqual(new_thresh.item(), 0.45, places=6)
        metric = main.Reporter.report("routing_adjustment")
        self.assertAlmostEqual(metric.item(), 0.5, places=6)

    def test_graph_forward_triggers_adjustment(self):
        graph = Graph(reporter=main.Reporter)
        n1 = Neuron(zero=self.zero, activation_threshold=torch.tensor(0.5, requires_grad=True))
        n2 = Neuron(zero=self.zero, activation_threshold=torch.tensor(0.5, requires_grad=True))
        s1 = Synapse(zero=self.zero)
        graph.add_neuron("n1", n1)
        graph.add_neuron("n2", n2)
        graph.add_synapse("s1", "n1", "n2", s1)
        n1.record_local_loss(torch.tensor(0.1))
        n2.record_local_loss(torch.tensor(0.2))
        result = graph.forward(global_loss_target=self.zero)
        print("Forward result:", result)
        influence = main.Reporter.report("routing_adjustment")
        self.assertIsNotNone(influence)
        self.assertTrue(
            n1.activation_threshold.item() != 0.5
            or n2.activation_threshold.item() != 0.5
        )


if __name__ == "__main__":
    unittest.main()
