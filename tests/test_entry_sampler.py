import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from network.entities import Neuron
from network.graph import Graph
from network.entry_sampler import EntrySampler


class CapturingReporter:
    def __init__(self):
        self.calls = []

    def report(self, *args):
        self.calls.append(args)


class TestEntrySampler(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.zero = torch.tensor(0.0)
        self.reporter = CapturingReporter()

    def test_sampling_distribution(self):
        graph = Graph(reporter=self.reporter)
        n1 = Neuron(zero=self.zero)
        n1.phi_v = torch.tensor(0.0)
        n2 = Neuron(zero=self.zero)
        n2.phi_v = torch.tensor(1.0)
        n3 = Neuron(zero=self.zero)
        n3.phi_v = torch.tensor(2.0)
        graph.add_neuron('a', n1)
        graph.add_neuron('b', n2)
        graph.add_neuron('c', n3)
        sampler = EntrySampler(temperature=torch.tensor(1.0), torch=torch, reporter=self.reporter)
        probs = sampler.compute_probabilities(graph)
        stacked = torch.stack([probs['a'], probs['b'], probs['c']])
        self.assertTrue(torch.allclose(torch.sum(stacked), torch.tensor(1.0)))
        chosen = sampler.sample_entry()
        self.assertIs(chosen, n3)
        self.assertIn(('entry_sample', 'c'), self.reporter.calls)


if __name__ == '__main__':
    unittest.main()
