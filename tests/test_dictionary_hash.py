import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.poted import PoTED


class TestDictionaryHash(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_roundtrip_without_dictionary_mismatch(self):
        engine = PoTED(reporter=main.Reporter)
        tensor = engine({'msg': 'foo'})
        result = engine.decode(tensor)
        print('Decoded result:', result)
        self.assertEqual(result, {'msg': 'foo'})


if __name__ == '__main__':
    unittest.main()
