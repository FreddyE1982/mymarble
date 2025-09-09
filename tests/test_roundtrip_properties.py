import unittest
import sys
import pathlib

tests_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(tests_dir))
sys.path.append(str(tests_dir.parent))

import main
from hypothesis import given, strategies as st
from utils import roundtrip

json_strategy = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda children: st.lists(children, max_size=5) | st.dictionaries(st.text(), children, max_size=5),
    max_leaves=20,
)


class TestRoundtripProperties(unittest.TestCase):
    @given(json_strategy)
    def test_roundtrip_identity(self, obj):
        main.Reporter._metrics = {}
        result = roundtrip(obj)
        print('Roundtrip object:', obj)
        self.assertEqual(result, obj)
        self.assertIsNone(main.Reporter.report('roundtrip_failures'))

    def test_roundtrip_failure_metrics(self):
        main.Reporter._metrics = {}
        with self.assertRaises(TypeError):
            roundtrip({1, 2, 3})
        self.assertEqual(main.Reporter.report('roundtrip_failures'), 1)
        print('Roundtrip failures:', main.Reporter.report('roundtrip_failures'))
