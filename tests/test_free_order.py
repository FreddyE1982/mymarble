import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main


class TestFreeOrder(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_free_non_lifo(self):
        device = main.MemoryDevice('VRAM', 100)
        a = device.allocate(30)
        b = device.allocate(20)
        c = device.allocate(10)
        start_b = b.start
        device.free(b)
        d = device.allocate(15)
        print('Blocks after reuse:', device.blocks)
        self.assertEqual(d.start, start_b)
        self.assertIn(a, device.allocations)
        self.assertIn(c, device.allocations)
        self.assertIn(d, device.allocations)
        self.assertEqual(device.used, 55)


if __name__ == '__main__':
    unittest.main()
