import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.poted import PoTED


class TestMultipleRoundtrips(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_out_of_order_decoding(self):
        engine = PoTED(reporter=main.Reporter)
        tensor_a = engine({'msg': 'A'})
        tensor_b = engine({'msg': 'B'})
        result_a = engine.decode(tensor_a)
        result_b = engine.decode(tensor_b)
        print('Decoded objects:', result_a, result_b)
        self.assertEqual(result_a, {'msg': 'A'})
        self.assertEqual(result_b, {'msg': 'B'})


if __name__ == '__main__':
    unittest.main()
