import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main


class TestCapacityGuard(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_budget_not_exceed_capacity(self):
        with self.assertRaises(ValueError):
            main.MemoryDevice('VRAM', 512, budget=1024)
        print('Invalid configs:', main.Reporter.report('invalid_device_configs'))
        self.assertEqual(main.Reporter.report('invalid_device_configs'), 1)

    def test_reserved_memory_counts_toward_capacity(self):
        device = main.MemoryDevice('VRAM', 100)
        device.allocate(60, reserve=30)
        print('Metrics after reserve:', main.Reporter.report(['VRAM_used', 'VRAM_reserved']))
        with self.assertRaises(main.DeviceCapacityExceeded):
            device.allocate(20)
        print('Metrics after failed alloc:', main.Reporter.report(['VRAM_used', 'VRAM_reserved']))
        self.assertEqual(device.used, 40)
        self.assertEqual(device.reserved, 30)


if __name__ == '__main__':
    unittest.main()
