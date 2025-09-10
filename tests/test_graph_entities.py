import unittest
import sys
import pathlib
import time
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from network.entities import Neuron, Synapse
from network.graph import Graph
from network.latency import LatencyEstimator


class TestGraphEntities(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}
        self.zero = torch.tensor(0.0)

    def test_tensor_states_and_latency(self):
        estimator = LatencyEstimator(reporter=main.Reporter, zero=self.zero)
        graph = Graph(latency_estimator=estimator, reporter=main.Reporter)
        n1 = Neuron(zero=self.zero)
        n2 = Neuron(zero=self.zero)
        s1 = Synapse(zero=self.zero)
        graph.add_neuron("n1", n1)
        graph.add_neuron("n2", n2)
        graph.add_synapse("s1", "n1", "n2", s1)
        self.assertEqual(n1.lambda_v, self.zero)
        self.assertEqual(s1.lambda_e, self.zero)
        graph.forward(global_loss_target=self.zero)
        time.sleep(0.01)
        graph.forward(global_loss_target=self.zero)
        print("Neuron latency after update:", n1.lambda_v)
        print("Synapse latency after update:", s1.lambda_e)
        self.assertGreater(n1.lambda_v.item(), 0)
        self.assertGreater(s1.lambda_e.item(), 0)
        snapshot = n1.to_dict()
        print("Neuron state snapshot:", snapshot)
        self.assertIn("activation", snapshot)


if __name__ == "__main__":
    unittest.main()
