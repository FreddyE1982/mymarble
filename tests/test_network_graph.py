import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from network.entities import Neuron, Synapse
from network.graph import Graph


class TestNetworkGraph(unittest.TestCase):
    def test_graph_multiedges_and_removals(self):
        zero = torch.tensor(0)
        g = Graph()
        n1 = Neuron(zero=zero)
        n2 = Neuron(zero=zero)
        g.add_neuron('n1', n1)
        g.add_neuron('n2', n2)
        s1 = Synapse(zero=zero)
        s2 = Synapse(zero=zero)
        g.add_synapse('s1', 'n1', 'n2', s1)
        g.add_synapse('s2', 'n1', 'n2', s2)
        self.assertEqual(len(g.get_synapses('n1', 'n2')), 2)
        g.remove_synapse('s1')
        self.assertEqual(len(g.get_synapses('n1', 'n2')), 1)
        g.remove_neuron('n1')
        self.assertIsNone(g.get_neuron('n1'))
        self.assertEqual(len(g.synapses), 0)

    def test_default_zero_tensors(self):
        zero = torch.tensor(0)
        neuron = Neuron(zero=zero)
        synapse = Synapse(zero=zero)
        for key, value in neuron.to_dict().items():
            if key == "weight":
                self.assertEqual(value, 1.0)
            else:
                self.assertTrue(torch.equal(value, zero))
        for key, value in synapse.to_dict().items():
            if key == "weight":
                self.assertEqual(value, 1.0)
            else:
                self.assertTrue(torch.equal(value, zero))


if __name__ == '__main__':
    unittest.main()
