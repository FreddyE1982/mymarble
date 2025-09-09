import unittest
import sys
import pathlib
import numpy as np
import torch
from hypothesis import given, strategies as st, settings

tests_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(tests_dir))
sys.path.append(str(tests_dir.parent))

import main
from poted.poted import PoTED

json_strategy = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda children: st.lists(children, max_size=5) | st.dictionaries(st.text(), children, max_size=5),
    max_leaves=20,
)


class TestPoTEDEndToEnd(unittest.TestCase):
    def setUp(self):
        self.engine = PoTED(reporter=main.Reporter)

    def _roundtrip(self, obj):
        main.Reporter._metrics = {}
        tensor = self.engine(obj)
        result = self.engine.decode(tensor)
        print('Roundtrip input:', obj)
        print('Roundtrip tensor:', tensor)
        print('Roundtrip result:', result)
        total = main.Reporter.report('total_tokens')
        ratio = main.Reporter.report('compression_ratio')
        shape = main.Reporter.report('tensor_shape')
        print('Reporter metrics total_tokens:', total)
        print('Reporter metrics compression_ratio:', ratio)
        print('Reporter metrics tensor_shape:', shape)
        self.assertIsNotNone(total)
        self.assertIsNotNone(ratio)
        self.assertIsNotNone(shape)
        return result

    def test_roundtrip_dict(self):
        obj = {'a': 1, 'b': [1, 2, 3]}
        result = self._roundtrip(obj)
        self.assertEqual(result, obj)

    def test_roundtrip_list(self):
        obj = [1, 2, {'c': 3}]
        result = self._roundtrip(obj)
        self.assertEqual(result, obj)

    def test_roundtrip_nested(self):
        obj = {'x': [1, {'y': [2, 3]}, 4]}
        result = self._roundtrip(obj)
        self.assertEqual(result, obj)

    def test_numpy_tensor_error(self):
        np_array = np.array([1, 2, 3])
        with self.assertRaises(TypeError):
            self.engine(np_array)

    def test_torch_tensor_error(self):
        torch_tensor = torch.tensor([1, 2, 3])
        with self.assertRaises(TypeError):
            self.engine(torch_tensor)

    @settings(max_examples=10)
    @given(json_strategy)
    def test_property_roundtrip(self, obj):
        result = self._roundtrip(obj)
        self.assertEqual(result, obj)
