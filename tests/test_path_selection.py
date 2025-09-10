import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from network.entities import Neuron, Synapse
from network.path_selector import PathSelector


class TestPathSelection(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}
        self.zero = torch.tensor(0.0)

    def test_minimal_loss_path(self):
        neuron = Neuron(zero=self.zero)
        neuron.record_local_loss(torch.tensor(2.0))
        s1 = Synapse(lambda_e=torch.tensor(1.0), c_e=torch.tensor(0.0), zero=self.zero)
        s2 = Synapse(lambda_e=torch.tensor(0.1), c_e=torch.tensor(0.0), zero=self.zero)
        selector = PathSelector(reporter=main.Reporter)
        state = {"outgoing_synapses": [s1, s2], "global_loss_target": self.zero}
        chosen = selector.select_path(neuron, state)
        print("Path selector calls:", main.Reporter.report("path_selector_calls"))
        print("Last path score:", main.Reporter.report("path_selector_last_score"))
        self.assertIs(chosen, s2)


if __name__ == "__main__":
    unittest.main()
