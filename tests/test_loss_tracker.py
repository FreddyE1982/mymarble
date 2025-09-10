import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from network.entities import Neuron
from network.learning import LossTracker


class TestLossTracker(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_loss_tracking(self):
        reporter = main.Reporter
        neuron = Neuron()
        tracker = LossTracker(reporter)
        tracker.update_loss(neuron, [1, 3])
        self.assertEqual(neuron.last_local_loss, 2)
        tracker.update_loss(neuron, [3, 5])
        self.assertEqual(neuron.last_local_loss, 3)
        self.assertEqual(reporter.report('loss_updates'), 2)
        print('Stats after updates:', tracker.get_stats(neuron))
        print('Loss updates metric:', reporter.report('loss_updates'))

    def test_loss_tracking_detach(self):
        reporter = main.Reporter
        neuron = Neuron()
        tracker = LossTracker(reporter, zero=torch.tensor(0.0))
        losses = [
            torch.tensor(1.0, requires_grad=True),
            torch.tensor(3.0, requires_grad=True),
        ]
        avg = tracker.update_loss(neuron, losses)
        print('Detached average:', avg)
        print('Neuron stored loss requires_grad:', neuron.last_local_loss.requires_grad)
        self.assertFalse(avg.requires_grad)
        self.assertFalse(neuron.last_local_loss.requires_grad)

    def test_eq_01_updates(self):
        reporter = main.Reporter
        neuron = Neuron()
        tracker = LossTracker(reporter, zero=torch.tensor(0.0))
        tracker.update_loss(neuron, [torch.tensor(1.0), torch.tensor(3.0)])
        tracker.update_loss(neuron, [torch.tensor(3.0), torch.tensor(5.0)])
        stats = tracker.get_stats(neuron)
        print('Tracker stats:', stats)
        self.assertEqual(stats['t'], 2)
        self.assertEqual(stats['m'], 4)
        self.assertAlmostEqual(stats['avg'].item(), 3.0, places=5)


if __name__ == '__main__':
    unittest.main()
