import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.pipeline import TensorBuilder


class TestTensorShapes(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def _run_case(self, Lw, Le, Lu, tokens, expected_shape):
        builder = TensorBuilder(Lw=Lw, Le=Le, Lu=Lu, reporter=main.Reporter)
        tensor = builder.to_tensor(tokens)
        self.assertEqual(list(tensor.shape), expected_shape)
        self.assertEqual(main.Reporter.report('tensor_shape'), expected_shape)
        print('Parameters:', Lw, Le, Lu, '-> shape:', tensor.shape)

    def test_various_shapes(self):
        cases = [
            (2, 3, 1, [1, 2, 3, 4], [1, 6]),
            (1, 1, 1, [1], [1, 3]),
            (5, 0, 0, [1, 2, 3, 4, 5, 6, 7], [1, 5]),
            (2, 0, 0, [[1], [1, 2, 3]], [2, 2]),
        ]
        for Lw, Le, Lu, tokens, shape in cases:
            with self.subTest(Lw=Lw, Le=Le, Lu=Lu, tokens=tokens):
                self._run_case(Lw, Le, Lu, tokens, shape)


if __name__ == '__main__':
    unittest.main()
