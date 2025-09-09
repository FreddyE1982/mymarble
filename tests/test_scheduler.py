import unittest
import main

class TestScheduler(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
