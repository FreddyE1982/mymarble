import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main


class TestFragmentation(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_compaction_reduces_fragmentation(self):
        device = main.MemoryDevice('VRAM', 100)
        device.allocate(30)
        device.allocate(30)
        device.free(30)
        device.allocate(10)
        device.allocate(10)
        device.free(10)
        device.free(10)
        before = main.Reporter.report(['VRAM_fragmentation_ratio', 'VRAM_largest_free_block'])
        print('Before compaction:', before)
        with self.assertRaises(main.DeviceCapacityExceeded):
            device.allocate(50)
        self.assertEqual(main.Reporter.report('allocation_failures'), 1)
        device.compact()
        after = main.Reporter.report(['VRAM_fragmentation_ratio', 'VRAM_largest_free_block'])
        print('After compaction:', after)
        self.assertLess(after[0], before[0])
        device.allocate(50)
        self.assertEqual(main.Reporter.report('allocation_failures'), 1)


if __name__ == '__main__':
    unittest.main()
