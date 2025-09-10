import unittest
import sys
import pathlib

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


if __name__ == '__main__':
    unittest.main()
