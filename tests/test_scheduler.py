import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main


class TestScheduler(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_scheduler_records_metrics(self):
        device = main.MemoryDevice('VRAM', 512)
        ops = [
            main.Operation('op1', 128, 1, device),
            main.Operation('op2', 128, 1, device),
        ]
        scheduler = main.Scheduler(device)
        duration = scheduler.run(ops)
        self.assertEqual(duration, 2)
        self.assertEqual(main.Reporter.report('VRAM_used'), 0)
        self.assertEqual(main.Reporter.report('makespan'), 2)

    def test_budget_overrun_records_metric(self):
        device = main.MemoryDevice('VRAM', 512, budget=256)
        ops = [main.Operation('op1', 300, 1, device)]
        scheduler = main.Scheduler(device)
        with self.assertRaises(main.DeviceBudgetExceeded):
            scheduler.run(ops)
        self.assertEqual(main.Reporter.report('budget_exceeded'), 1)


if __name__ == '__main__':
    unittest.main()
