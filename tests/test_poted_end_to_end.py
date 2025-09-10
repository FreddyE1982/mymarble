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
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: st.lists(children, max_size=5)
    | st.dictionaries(st.text(), children, max_size=5),
    max_leaves=20,
)


class TestPoTEDEndToEnd(unittest.TestCase):
    def setUp(self):
        self.engine = PoTED(reporter=main.Reporter)

    def _roundtrip(self, obj):
        main.Reporter._metrics = {}
        tensor = self.engine(obj)
        result = self.engine.decode(tensor)
        print("Roundtrip input:", obj)
        print("Roundtrip tensor:", tensor)
        print("Roundtrip result:", result)
        total = main.Reporter.report("total_tokens")
        ratio = main.Reporter.report("compression_ratio")
        shape = main.Reporter.report("tensor_shape")
        print("Reporter metrics total_tokens:", total)
        print("Reporter metrics compression_ratio:", ratio)
        print("Reporter metrics tensor_shape:", shape)
        self.assertIsNotNone(total)
        self.assertIsNotNone(ratio)
        self.assertIsNotNone(shape)
        return result

    def test_roundtrip_dict(self):
        obj = {"a": 1, "b": [1, 2, 3]}
        result = self._roundtrip(obj)
        self.assertEqual(result, obj)

    def test_roundtrip_list(self):
        obj = [1, 2, {"c": 3}]
        result = self._roundtrip(obj)
        self.assertEqual(result, obj)

    def test_roundtrip_nested(self):
        obj = {"x": [1, {"y": [2, 3]}, 4]}
        result = self._roundtrip(obj)
        self.assertEqual(result, obj)

    def test_roundtrip_numpy_array(self):
        np_array = np.array([1, 2, 3])
        result = self._roundtrip(np_array)
        self.assertTrue(np.array_equal(result, np_array))

    def test_roundtrip_torch_tensor(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_tensor = torch.tensor(
            [1.0, 2.0, 3.0], device=device, requires_grad=True
        )
        result = self._roundtrip(torch_tensor)
        self.assertTrue(torch.equal(result, torch_tensor))
        self.assertEqual(result.device, torch_tensor.device)
        self.assertEqual(result.requires_grad, torch_tensor.requires_grad)

    @settings(max_examples=10)
    @given(json_strategy)
    def test_property_roundtrip(self, obj):
        result = self._roundtrip(obj)
        self.assertEqual(result, obj)
