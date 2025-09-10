import unittest
import sys
import pathlib
import time
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from network.entities import Neuron, Synapse
from network.latency import LatencyEstimator


class TestLatencyReporting(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}
        self.zero = torch.tensor(0.0)

    def test_latency_metrics(self):
        estimator = LatencyEstimator(reporter=main.Reporter, zero=self.zero)
        neuron = Neuron(zero=self.zero)
        synapse = Synapse(zero=self.zero)
        # first update to initialise timestamps
        estimator.update("n1", neuron, {"s1": synapse})
        time.sleep(0.01)
        estimator.update("n1", neuron, {"s1": synapse})
        n_metric = main.Reporter.report("latency_neuron_n1")
        s_metric = main.Reporter.report("latency_synapse_s1")
        print("Reported neuron latency:", n_metric)
        print("Reported synapse latency:", s_metric)
        self.assertGreater(n_metric.item(), 0)
        self.assertGreater(s_metric.item(), 0)
        print("Synapse lambda and cost:", synapse.lambda_e, synapse.c_e)
        self.assertEqual(synapse.lambda_e, synapse.c_e)


if __name__ == "__main__":
    unittest.main()
