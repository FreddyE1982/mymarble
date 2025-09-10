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
        n1.phi_v = torch.tensor(1.0)
        n2 = Neuron(zero=self.zero)
        n2.phi_v = torch.tensor(-100.0)
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

    def test_cumulative_loss_tracking(self):
        n = Neuron(zero=self.zero)
        n.update_cumulative_loss(torch.tensor(1.5), reporter=main.Reporter)
        self.assertAlmostEqual(n.step_loss.item(), 1.5)
        self.assertAlmostEqual(n.cumulative_loss.item(), 1.5)
        step_metric = main.Reporter.report(f"neuron_{id(n)}_step_loss")
        cum_metric = main.Reporter.report(f"neuron_{id(n)}_cumulative_loss")
        print("Recorded step loss:", step_metric)
        print("Recorded cumulative loss:", cum_metric)
        self.assertAlmostEqual(step_metric.item(), 1.5)
        self.assertAlmostEqual(cum_metric.item(), 1.5)
        n.update_cumulative_loss(torch.tensor(2.0), reporter=main.Reporter)
        self.assertAlmostEqual(n.step_loss.item(), 2.0)
        self.assertAlmostEqual(n.cumulative_loss.item(), 3.5)
        step_metric = main.Reporter.report(f"neuron_{id(n)}_step_loss")
        cum_metric = main.Reporter.report(f"neuron_{id(n)}_cumulative_loss")
        print("Updated step loss:", step_metric)
        print("Updated cumulative loss:", cum_metric)
        self.assertAlmostEqual(step_metric.item(), 2.0)
        self.assertAlmostEqual(cum_metric.item(), 3.5)
        n.reset()
        self.assertAlmostEqual(n.step_loss.item(), 0.0)
        self.assertAlmostEqual(n.cumulative_loss.item(), 0.0)
        snapshot = n.to_dict()
        print("Neuron snapshot after reset:", snapshot)
        self.assertIn("step_loss", snapshot)
        self.assertIn("cumulative_loss", snapshot)

    def test_activation_recording(self):
        n = Neuron(zero=self.zero)
        n.update_cumulative_loss(torch.tensor(1.0), reporter=main.Reporter)
        n.record_activation(
            torch.tensor(1.0), torch.tensor(2.0), reporter=main.Reporter
        )
        self.assertAlmostEqual(n.timestamp.item(), 1.0)
        self.assertAlmostEqual(n.measured_time.item(), 2.0)
        speed_metric = main.Reporter.report(f"neuron_{id(n)}_loss_decrease_speed")
        print("Recorded loss decrease speed:", speed_metric)
        self.assertAlmostEqual(speed_metric.item(), 0.5)
        self.assertAlmostEqual(n.loss_decrease_speed.item(), 0.5)
        snapshot = n.to_dict()
        print("Activation snapshot:", snapshot)
        self.assertIn("timestamp", snapshot)
        self.assertIn("measured_time", snapshot)
        self.assertIn("loss_decrease_speed", snapshot)

    def test_loss_speed_accumulates_multiple_steps(self):
        n = Neuron(zero=self.zero)
        # establish initial activation so that previous timestamp and cumulative
        # loss are persisted
        n.record_activation(self.zero, self.zero, reporter=main.Reporter)
        # simulate two loss updates prior to the next activation
        n.update_cumulative_loss(torch.tensor(1.0), reporter=main.Reporter)
        n.update_cumulative_loss(torch.tensor(2.0), reporter=main.Reporter)
        n.record_activation(
            torch.tensor(3.0), torch.tensor(3.0), reporter=main.Reporter
        )
        expected_speed = (1.0 + 2.0) / 3.0
        speed_metric = main.Reporter.report(
            f"neuron_{id(n)}_loss_decrease_speed"
        )
        print("Accumulated loss decrease speed:", speed_metric)
        self.assertAlmostEqual(speed_metric.item(), expected_speed)
        self.assertAlmostEqual(n.loss_decrease_speed.item(), expected_speed)
        snapshot = n.to_dict()
        print("Snapshot after accumulated activation:", snapshot)
        self.assertIn("prev_cumulative_loss", snapshot)
        self.assertAlmostEqual(
            snapshot["prev_cumulative_loss"].item(), n.cumulative_loss.item()
        )


if __name__ == "__main__":
    unittest.main()
