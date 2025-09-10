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
        n1.last_local_loss = torch.tensor(1.0)
        n2.last_local_loss = torch.tensor(2.0)
        s1 = Synapse(zero=self.zero)
        s2 = Synapse(zero=self.zero)
        s1.gate = torch.tensor(0.7)
        s2.gate = torch.tensor(0.8)
        s1.c_e = torch.tensor(3.0)
        s2.c_e = torch.tensor(4.0)
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


if __name__ == "__main__":
    unittest.main()
