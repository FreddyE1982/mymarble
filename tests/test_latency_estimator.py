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


if __name__ == '__main__':
    unittest.main()
