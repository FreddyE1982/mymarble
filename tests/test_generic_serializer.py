import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.serializer import GenericSerializer


class TestGenericSerializer(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_basic_roundtrip(self):
        serializer = GenericSerializer(reporter=main.Reporter)
        data = [1, -2, 3.5, b'hi', {'a': 1, 'b': [2, 3]}, None, True, False]
        stream = serializer.serialize(data)
        print('Serialized stream length:', len(stream))
        result = serializer.deserialize(stream)
        print('Deserialized object:', result)
        self.assertEqual(result, data)

    def test_recursion_depth_reporting(self):
        serializer = GenericSerializer(reporter=main.Reporter)
        nested = {'lvl1': {'lvl2': {'lvl3': {'lvl4': {'lvl5': 5}}}}}
        serializer.serialize(nested)
        depth = main.Reporter.report('max_recursion_depth')
        print('Reported max recursion depth:', depth)
        self.assertEqual(depth, 6)
