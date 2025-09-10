import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from main import Reporter
from network.entities import Neuron
from network.path_forwarder import PathForwarder


class TestPathForwarder(unittest.TestCase):
    def test_run_updates_and_reports(self):
        Reporter._metrics = {}
        times = iter([0.0, 1.0, 3.0])

        def fake_time():
            return next(times)

        n1 = Neuron(zero=0)
        n2 = Neuron(zero=0)
        forwarder = PathForwarder(reporter=Reporter, time_source=fake_time)
        result = forwarder.run([n1, n2], [1.0, 2.0])
        print("result", result)
        print("metric", Reporter.report("path_total_time"))
        self.assertEqual(result["path_time"], 3.0)
        self.assertEqual(result["final_loss"], n2.cumulative_loss)
        self.assertEqual(Reporter.report("path_total_time"), 3.0)


if __name__ == "__main__":
    unittest.main()
