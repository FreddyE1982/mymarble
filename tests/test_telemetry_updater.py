import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from main import Reporter
from network.entities import Neuron
from network.telemetry import TelemetryUpdater


class TestTelemetryUpdater(unittest.TestCase):
    def setUp(self):
        Reporter._metrics = {}

    def test_hard_routing_updates(self):
        times = iter([0.0, 1.0])

        def fake_time():
            return next(times)

        n1 = Neuron(zero=torch.tensor(0.0))
        n2 = Neuron(zero=torch.tensor(0.0))
        n1.update_latency(torch.tensor(0.5))
        n2.update_latency(torch.tensor(1.5))

        updater = TelemetryUpdater(reporter=Reporter, time_source=fake_time, ema_alpha=0.5)
        losses = [torch.tensor(1.0), torch.tensor(2.0)]
        result = updater.update([n1, n2], losses)

        print("Reporter metrics:", Reporter._metrics)
        print("n1 step_time, speed:", n1.step_time, n1.loss_decrease_speed)
        print("n2 step_time, speed:", n2.step_time, n2.loss_decrease_speed)
        print("final_loss:", result["final_loss"])

        self.assertAlmostEqual(n1.step_time.item(), 0.25)
        self.assertAlmostEqual(n2.step_time.item(), 0.75)
        self.assertEqual(n1.cumulative_loss.item(), 1.0)
        self.assertEqual(n2.cumulative_loss.item(), 2.0)
        self.assertEqual(Reporter.report(f"neuron_{id(n1)}_step_time").item(), 0.25)
        self.assertEqual(Reporter.report(f"neuron_{id(n2)}_step_time").item(), 0.75)
        self.assertEqual(Reporter.report(f"neuron_{id(n2)}_loss_decrease_speed").item(), 2.0)

    def test_soft_routing_weighting(self):
        times = iter([0.0, 1.0])

        def fake_time():
            return next(times)

        n1 = Neuron(zero=torch.tensor(0.0))
        n2 = Neuron(zero=torch.tensor(0.0))
        n1.update_latency(torch.tensor(2.0))
        n2.update_latency(torch.tensor(4.0))

        updater = TelemetryUpdater(reporter=Reporter, time_source=fake_time, ema_alpha=0.5)
        losses = [torch.tensor(2.0), torch.tensor(4.0)]
        g_v = {n1: 0.5, n2: 0.25}
        updater.update([n1, n2], losses, g_v=g_v)

        print("Reporter metrics:", Reporter._metrics)
        print("n1 step_time, speed:", n1.step_time, n1.loss_decrease_speed)
        print("n2 step_time, speed:", n2.step_time, n2.loss_decrease_speed)

        self.assertAlmostEqual(n1.step_time.item(), 0.5)
        self.assertAlmostEqual(n2.step_time.item(), 0.5)
        self.assertEqual(n1.cumulative_loss.item(), 1.0)
        self.assertEqual(n2.cumulative_loss.item(), 1.0)
        self.assertEqual(Reporter.report(f"neuron_{id(n1)}_loss_decrease_speed").item(), 0.0)
        self.assertEqual(Reporter.report(f"neuron_{id(n2)}_loss_decrease_speed").item(), 1.0)


if __name__ == "__main__":
    unittest.main()
