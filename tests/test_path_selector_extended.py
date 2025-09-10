import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from network.entities import Neuron
from network.path_selector import PathSelector


class DummyPath:
    def __init__(self, name, cost):
        self.name = name
        self.cost = torch.tensor(cost)


class TestExtendedPathSelector(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}
        self.neuron = Neuron()
        self.selector = PathSelector(reporter=main.Reporter)

    def test_select_exact(self):
        paths = [DummyPath('a', 3.0), DummyPath('b', 1.0), DummyPath('c', 2.0)]
        best, sampled = self.selector.select_exact(self.neuron, paths)
        print('selected_path_metric', main.Reporter.report(f'neuron_{id(self.neuron)}_selected_path'))
        print('sampled_cost_metric', main.Reporter.report(f'neuron_{id(self.neuron)}_sampled_path_cost'))
        self.assertIs(best, paths[1])
        self.assertIs(sampled, paths[1])

    def test_select_soft(self):
        torch.manual_seed(2)
        paths = [DummyPath('a', 0.1), DummyPath('b', 2.0)]
        best, sampled = self.selector.select_soft(self.neuron, paths, torch.tensor(0.0), torch.tensor(1.0))
        print('selected_path_metric', main.Reporter.report(f'neuron_{id(self.neuron)}_selected_path'))
        print('sampled_cost_metric', main.Reporter.report(f'neuron_{id(self.neuron)}_sampled_path_cost'))
        self.assertIs(best, paths[0])
        self.assertIs(sampled, paths[1])


if __name__ == '__main__':
    unittest.main()
