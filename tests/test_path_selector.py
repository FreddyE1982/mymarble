import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from network.entities import Neuron, Synapse
from network.path_selector import PathSelector
from network.graph import Graph


class TestPathSelector(unittest.TestCase):
    def setUp(self):
        self.zero = torch.tensor(0.0)

    def _build_graph(self, selector, synapses):
        g = Graph(path_selector=selector)
        n1 = Neuron(zero=self.zero)
        n1.record_local_loss(self.zero)
        n2 = Neuron(zero=self.zero)
        g.add_neuron('n1', n1)
        g.add_neuron('n2', n2)
        for idx, syn in enumerate(synapses, start=1):
            g.add_synapse(f's{idx}', 'n1', 'n2', syn)
        return g

    def test_select_lowest_score(self):
        selector = PathSelector()
        s1 = Synapse(lambda_e=torch.tensor(5.0), c_e=torch.tensor(0.0), zero=self.zero)
        s2 = Synapse(lambda_e=torch.tensor(1.0), c_e=torch.tensor(10.0), zero=self.zero)
        g = self._build_graph(selector, [s1, s2])
        result = g.forward(global_loss_target=self.zero)
        self.assertIs(result['n1'], s1)

    def test_latency_weighting(self):
        selector = PathSelector(latency_weight=10.0)
        s1 = Synapse(lambda_e=torch.tensor(5.0), c_e=torch.tensor(0.0), zero=self.zero)
        s2 = Synapse(lambda_e=torch.tensor(1.0), c_e=torch.tensor(10.0), zero=self.zero)
        g = self._build_graph(selector, [s1, s2])
        result = g.forward(global_loss_target=self.zero)
        self.assertIs(result['n1'], s2)


if __name__ == '__main__':
    unittest.main()
