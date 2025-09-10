import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main


class DummyTensor:
    def __init__(self, size, device):
        self.nbytes = size
        self.device = device


class TestScheduler(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_scheduler_records_metrics(self):
        device = main.MemoryDevice('VRAM', 512)
        ops = [
            main.Operation('op1', 128, 1, 'VRAM'),
            main.Operation('op2', 128, 1, 'VRAM'),
        ]
        scheduler = main.TensorLoadBalancer([device])
        duration = scheduler.run(ops)
        print('Sequential metrics:', main.Reporter.report(['VRAM_used', 'makespan', 'VRAM_idle_time']))
        self.assertEqual(duration, 2)
        self.assertEqual(main.Reporter.report('VRAM_used'), 0)
        self.assertEqual(main.Reporter.report('makespan'), 2)

    def test_budget_overrun_records_metric(self):
        device = main.MemoryDevice('VRAM', 512, budget=256)
        ops = [main.Operation('op1', 300, 1, 'VRAM')]
        scheduler = main.TensorLoadBalancer([device])
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
        scheduler = main.TensorLoadBalancer([gpu, cpu])
        duration = scheduler.run_parallel(ops)
        print('Parallel metrics:', main.Reporter.report(['makespan', 'GPU_idle_time', 'CPU_idle_time']))
        self.assertEqual(duration, 4)
        self.assertEqual(main.Reporter.report('GPU_idle_time'), 0)
        self.assertEqual(main.Reporter.report('CPU_idle_time'), 0)
        self.assertEqual(main.Reporter.report('makespan'), 4)

    def test_register_unregister(self):
        device = main.MemoryDevice('VRAM', 256)
        balancer = main.TensorLoadBalancer([device])
        tensor = DummyTensor(128, 'VRAM')
        self.assertFalse(balancer.isRegistered(tensor))
        balancer.register(tensor)
        print('Post register used:', main.Reporter.report('VRAM_used'))
        self.assertTrue(balancer.isRegistered(tensor))
        self.assertEqual(main.Reporter.report('VRAM_used'), 128)
        balancer.unregister(tensor)
        print('Post unregister used:', main.Reporter.report('VRAM_used'))
        self.assertFalse(balancer.isRegistered(tensor))
        self.assertEqual(main.Reporter.report('VRAM_used'), 0)

    def test_duplicate_registration_rejected(self):
        d1 = main.MemoryDevice('VRAM', 256)
        d2 = main.MemoryDevice('cpu', 256)
        balancer = main.TensorLoadBalancer([d1, d2])
        tensor = DummyTensor(64, 'VRAM')
        balancer.register(tensor)
        tensor.device = 'cpu'
        balancer.register(tensor)
        registered_device = balancer._registry[id(tensor)]['device'].name
        print('Device after reregister:', registered_device)
        self.assertEqual(registered_device, 'cpu')

    def test_tensor_auto_registration(self):
        device = main.MemoryDevice('cpu', 256)
        balancer = main.TensorLoadBalancer([device])
        original_new = main.patch_tensor_registration(balancer)
        tensor = torch.Tensor(2, 2)
        print('Auto registration metric:', main.Reporter.report('registered_tensors'))
        self.assertTrue(tensor.isRegistered())
        tensor.unregister()
        print('Post auto unregister metric:', main.Reporter.report('registered_tensors'))
        self.assertFalse(tensor.isRegistered())
        torch.Tensor.__new__ = original_new


if __name__ == '__main__':
    unittest.main()
