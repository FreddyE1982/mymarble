import unittest
import torch
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from network.graph import Graph
from network.entities import Neuron, Synapse
from network.complexity import ComplexityCalculator


class TestComplexityCalculator(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_compute_complexity(self):
        graph = Graph(reporter=main.Reporter)
        n1 = Neuron()
        n2 = Neuron()
        graph.add_neuron('n1', n1)
        graph.add_neuron('n2', n2)
        s1 = Synapse()
        graph.add_synapse('s1', 'n1', 'n2', s1)
        weights = [torch.tensor([1.0, -1.0], requires_grad=True)]
        A = [torch.tensor(2.0)]
        B = [torch.tensor(3.0)]
        gamma = torch.tensor(0.5)
        lambda_ = torch.tensor(0.5)
        attrs = [{"a", "b"}]
        calc = ComplexityCalculator(A, B, gamma, lambda_, weights, attrs, main.Reporter)
        complexity = calc.compute(graph)
        expected = (
            torch.tensor(0.0)
            + 2
            + A[0] * 2
            + B[0] * 1
            + gamma * weights[0].abs().sum()
            + lambda_ * weights[0].pow(2).sum().sqrt()
            + 2
        )
        self.assertTrue(torch.allclose(complexity, expected))
        self.assertEqual(main.Reporter.report('num_neurons'), 2)
        self.assertIsNotNone(main.Reporter.report('edge_penalty'))
        self.assertIsNotNone(main.Reporter.report('l1_norm'))
        self.assertIsNotNone(main.Reporter.report('l2_norm'))
        self.assertEqual(main.Reporter.report('attribute_size'), 2)


if __name__ == '__main__':
    unittest.main()
