import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from network.entities import Neuron, Synapse
from network.path_cost import PathCostCalculator


class TestPathCost(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        main.Reporter._metrics = {}
        self.zero = torch.tensor(0.0)
        self.neuron = Neuron(
            last_local_loss=torch.tensor(2.0),
            lambda_v=torch.tensor(3.0),
            zero=self.zero,
        )
        self.synapse = Synapse(
            c_e=torch.tensor(5.0),
            lambda_e=torch.tensor(7.0),
            zero=self.zero,
        )
        self.calc = PathCostCalculator(reporter=main.Reporter, zero=self.zero)

    def test_compute_cost(self):
        cost = self.calc.compute_cost(
            [self.neuron, self.synapse],
            lambda_0=torch.tensor(1.0),
            lambda_max=torch.tensor(10.0),
            alpha=torch.tensor(0.5),
            beta=torch.tensor(2.0),
            T_heat=torch.tensor(5.0),
        )
        loss_metric = main.Reporter.report('path_loss')
        latency_metric = main.Reporter.report('path_latency')
        cost_metric = main.Reporter.report('path_cost')
        print('Computed loss:', loss_metric)
        print('Computed latency:', latency_metric)
        print('Computed cost:', cost_metric)
        loss = torch.tensor(7.0)
        latency = torch.tensor(10.0)
        delta = torch.clamp(latency - torch.tensor(1.0), min=0)
        denom = (torch.tensor(10.0) - torch.tensor(1.0)) + 1
        norm = delta / denom
        annealed = torch.log1p(norm * torch.tensor(5.0)) / torch.log1p(torch.tensor(5.0))
        expected_cost = torch.tensor(0.5) * loss + torch.tensor(2.0) * annealed
        self.assertTrue(torch.allclose(cost, expected_cost))
        self.assertTrue(torch.allclose(loss_metric, loss))
        self.assertTrue(torch.allclose(latency_metric, latency))
        self.assertTrue(torch.allclose(cost_metric, expected_cost))


if __name__ == '__main__':
    unittest.main()
