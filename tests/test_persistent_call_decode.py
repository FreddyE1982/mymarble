import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.poted import PoTED


class TestPersistentCallDecode(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_call_roundtrip(self):
        engine = PoTED(reporter=main.Reporter, persistent=True)
        tensor = engine({'msg': 'x'})
        result = engine.decode(tensor)
        print('Call decode result:', result)
        self.assertEqual(result, {'msg': 'x'})


if __name__ == '__main__':
    unittest.main()
