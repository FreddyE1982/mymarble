import unittest
import sys
import pathlib
import random
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from main import Reporter
from network.entities import Neuron
from network.path_forwarder import PathForwarder


class TestForwardPass(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        torch.manual_seed(0)
        Reporter._metrics = {}

    def test_two_neuron_path_metrics(self):
        times = iter([0.0, 1.0, 3.0])

        def fake_time():
            return next(times)

        n1 = Neuron(zero=0)
        n2 = Neuron(zero=0)
        forwarder = PathForwarder(reporter=Reporter, time_source=fake_time)
        losses = [1.0, 2.0]
        result = forwarder.run([n1, n2], losses)

        total_cumulative = n1.cumulative_loss + n2.cumulative_loss
        manual_sum = sum(losses)
        timestamps = [0.0, 1.0, 3.0]
        hop_durations = [timestamps[1] - timestamps[0], timestamps[2] - timestamps[1]]
        expected_path_time = sum(hop_durations)
        expected_speed_n1 = losses[0] / timestamps[1]
        expected_speed_n2 = losses[1] / timestamps[2]

        print("Reporter metrics:", Reporter._metrics)
        print(
            "n1 step,cumul,speed:",
            n1.step_loss,
            n1.cumulative_loss,
            n1.loss_decrease_speed,
        )
        print(
            "n2 step,cumul,speed:",
            n2.step_loss,
            n2.cumulative_loss,
            n2.loss_decrease_speed,
        )
        print("path_time result:", result["path_time"])

        self.assertEqual(total_cumulative, manual_sum)
        self.assertEqual(result["path_time"], expected_path_time)
        self.assertEqual(Reporter.report("path_total_time"), expected_path_time)
        self.assertEqual(result["final_loss"], n2.cumulative_loss)
        self.assertEqual(Reporter.report(f"neuron_{id(n1)}_step_loss"), losses[0])
        self.assertEqual(
            Reporter.report(f"neuron_{id(n1)}_cumulative_loss"), losses[0]
        )
        self.assertAlmostEqual(
            Reporter.report(f"neuron_{id(n1)}_loss_decrease_speed"),
            expected_speed_n1,
            places=6,
        )
        self.assertEqual(Reporter.report(f"neuron_{id(n2)}_step_loss"), losses[1])
        self.assertEqual(
            Reporter.report(f"neuron_{id(n2)}_cumulative_loss"), losses[1]
        )
        self.assertAlmostEqual(
            Reporter.report(f"neuron_{id(n2)}_loss_decrease_speed"),
            expected_speed_n2,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
