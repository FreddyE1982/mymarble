import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.poted import PoTED


class TestPersistentPoTED(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_sequential_decoding(self):
        engine = PoTED(reporter=main.Reporter, persistent=True)
        tensor_a = engine.encode({'msg': 'A'})
        tensor_b = engine.encode({'msg': 'B'})
        result_a = engine.decode(tensor_a)
        result_b = engine.decode(tensor_b)
        print('Sequential decode results:', result_a, result_b)
        self.assertEqual(result_a, {'msg': 'A'})
        self.assertEqual(result_b, {'msg': 'B'})


if __name__ == '__main__':
    unittest.main()
