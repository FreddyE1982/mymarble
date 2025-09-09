import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main


class TestEvictionPolicies(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_drop_policy_evicts_and_counts(self):
        device = main.MemoryDevice('VRAM', 100)
        device.allocate(80)
        with self.assertRaises(main.DeviceCapacityExceeded):
            device.allocate(50)
        print('Drop metrics:', main.Reporter.report(['VRAM_used', 'evictions']))
        self.assertEqual(device.used, 30)
        self.assertEqual(main.Reporter.report('evictions'), 1)

    def test_remap_policy_moves_and_counts(self):
        device = main.MemoryDevice('VRAM', 100, eviction_policy=main.RemapPolicy)
        device.allocate(80)
        with self.assertRaises(main.DeviceCapacityExceeded):
            device.allocate(50)
        print('Remap metrics:', main.Reporter.report(['VRAM_used', 'VRAM_reserved', 'remaps']))
        self.assertEqual(device.used, 30)
        self.assertEqual(device.reserved, 50)
        self.assertEqual(main.Reporter.report('remaps'), 1)


if __name__ == '__main__':
    unittest.main()
