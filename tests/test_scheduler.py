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
            main.Operation('op1', 128, 1, 'VRAM'),
            main.Operation('op2', 128, 1, 'VRAM'),
        ]
        scheduler = main.Scheduler([device])
        duration = scheduler.run(ops)
        print('Sequential metrics:', main.Reporter.report(['VRAM_used', 'makespan', 'VRAM_idle_time']))
        self.assertEqual(duration, 2)
        self.assertEqual(main.Reporter.report('VRAM_used'), 0)
        self.assertEqual(main.Reporter.report('makespan'), 2)

    def test_budget_overrun_records_metric(self):
        device = main.MemoryDevice('VRAM', 512, budget=256)
        ops = [main.Operation('op1', 300, 1, 'VRAM')]
        scheduler = main.Scheduler([device])
        with self.assertRaises(main.DeviceBudgetExceeded):
            scheduler.run(ops)
        print('Budget metric:', main.Reporter.report('budget_exceeded'))
        self.assertEqual(main.Reporter.report('budget_exceeded'), 1)

    def test_parallel_execution_reduces_makespan(self):
        gpu = main.MemoryDevice('GPU', 512)
        cpu = main.MemoryDevice('CPU', 512)
        ops = [
            main.Operation('gpu_op', 128, 4, 'GPU'),
            main.Operation('cpu_op', 128, 4, 'CPU'),
        ]
        scheduler = main.Scheduler([gpu, cpu])
        duration = scheduler.run_parallel(ops)
        print('Parallel metrics:', main.Reporter.report(['makespan', 'GPU_idle_time', 'CPU_idle_time']))
        self.assertEqual(duration, 4)
        self.assertEqual(main.Reporter.report('GPU_idle_time'), 0)
        self.assertEqual(main.Reporter.report('CPU_idle_time'), 0)
        self.assertEqual(main.Reporter.report('makespan'), 4)


if __name__ == '__main__':
    unittest.main()
