import unittest
import sys
import pathlib
import time
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from network.entities import Neuron, Synapse
from network.graph import Graph


class TestLatencyEstimator(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}
        self.zero = torch.tensor(0.0)

    def test_latency_reporting(self):
        graph = Graph(reporter=main.Reporter)
        n1 = Neuron(zero=self.zero)
        n2 = Neuron(zero=self.zero)
        graph.add_neuron('n1', n1)
        graph.add_neuron('n2', n2)
        s1 = Synapse(zero=self.zero)
        graph.add_synapse('s1', 'n1', 'n2', s1)
        n1.phi_v = self.zero
        n2.phi_v = torch.full_like(self.zero, float('-inf'))
        graph.forward(global_loss_target=self.zero)
        time.sleep(0.01)
        graph.forward(global_loss_target=self.zero)
        lv = n1.lambda_v
        le = s1.lambda_e
        print('Neuron latency:', lv)
        print('Synapse latency:', le)
        self.assertGreater(lv, 0)
        self.assertGreater(le, 0)
        metric_v = main.Reporter.report('latency_neuron_n1')
        metric_e = main.Reporter.report('latency_synapse_s1')
        print('Reported neuron latency:', metric_v)
        print('Reported synapse latency:', metric_e)
        self.assertEqual(lv, metric_v)
        self.assertEqual(le, metric_e)

    def test_inactive_neuron_no_latency_update(self):
        graph = Graph(reporter=main.Reporter)
        n1 = Neuron(zero=self.zero)
        n2 = Neuron(zero=self.zero)
        n3 = Neuron(zero=self.zero)
        graph.add_neuron('n1', n1)
        graph.add_neuron('n2', n2)
        graph.add_neuron('n3', n3)
        s1 = Synapse(zero=self.zero)
        graph.add_synapse('s1', 'n1', 'n2', s1)
        n1.phi_v = self.zero
        neg_inf = torch.full_like(self.zero, float('-inf'))
        n2.phi_v = neg_inf
        n3.phi_v = neg_inf
        graph.forward(global_loss_target=self.zero)
        time.sleep(0.01)
        graph.forward(global_loss_target=self.zero)
        print('Inactive neuron latency:', n3.lambda_v)
        self.assertEqual(n3.lambda_v, self.zero)
        metric = main.Reporter.report('latency_neuron_n3')
        print('Reported inactive neuron latency:', metric)
        self.assertIsNone(metric)

    def test_path_cost_uses_current_latency(self):
        graph = Graph(reporter=main.Reporter)
        n1 = Neuron(zero=self.zero)
        n2 = Neuron(zero=self.zero)
        graph.add_neuron('n1', n1)
        graph.add_neuron('n2', n2)
        s1 = Synapse(zero=self.zero)
        graph.add_synapse('s1', 'n1', 'n2', s1)
        n1.phi_v = self.zero
        n2.phi_v = torch.full_like(self.zero, float('-inf'))
        graph.forward(global_loss_target=self.zero)
        lat1 = main.Reporter.report('path_latency')
        print('Initial path latency:', lat1)
        time.sleep(0.05)
        graph.forward(global_loss_target=self.zero)
        lat2 = main.Reporter.report('path_latency')
        print('Path latency after wait:', lat2)
        self.assertGreater(lat2, lat1)


if __name__ == '__main__':
    unittest.main()
