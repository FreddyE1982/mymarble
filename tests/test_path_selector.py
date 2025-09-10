import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from network.entities import Neuron, Synapse
from network.path_selector import PathSelector


class TestPathSelector(unittest.TestCase):
    def setUp(self):
        self.zero = torch.tensor(0.0)
        self.neuron = Neuron(zero=self.zero)
        self.neuron.record_local_loss(self.zero)

    def test_select_lowest_score(self):
        selector = PathSelector()
        s1 = Synapse(lambda_e=torch.tensor(5.0), c_e=torch.tensor(0.0), zero=self.zero)
        s2 = Synapse(lambda_e=torch.tensor(1.0), c_e=torch.tensor(10.0), zero=self.zero)
        state = {"outgoing_synapses": [s1, s2], "global_loss_target": self.zero}
        result = selector.select_path(self.neuron, state)
        print('Selected synapse for lowest score:', 's1' if result is s1 else 's2')
        self.assertIs(result, s1)

    def test_latency_weighting(self):
        selector = PathSelector(latency_weight=10.0)
        s1 = Synapse(lambda_e=torch.tensor(5.0), c_e=torch.tensor(0.0), zero=self.zero)
        s2 = Synapse(lambda_e=torch.tensor(1.0), c_e=torch.tensor(10.0), zero=self.zero)
        state = {"outgoing_synapses": [s1, s2], "global_loss_target": self.zero}
        result = selector.select_path(self.neuron, state)
        print('Selected synapse with latency weighting:', 's1' if result is s1 else 's2')
        self.assertIs(result, s2)


if __name__ == '__main__':
    unittest.main()
