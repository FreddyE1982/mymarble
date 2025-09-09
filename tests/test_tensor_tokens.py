import unittest
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.tensor import TensorBuilder


class TestTensorBuilderToTokens(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_1d_tensor(self):
        builder = TensorBuilder(reporter=main.Reporter)
        tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        tokens = builder.to_tokens(tensor)
        print('Tokens from 1D tensor:', tokens)
        print('Extracted tokens metric:', main.Reporter.report('extracted_tokens'))
        self.assertEqual(tokens, [1, 2, 3])

    def test_2d_tensor(self):
        builder = TensorBuilder(reporter=main.Reporter)
        tensor = torch.tensor([[1, 2, 3], [4, 5, builder.PAD]], dtype=torch.int64)
        tokens = builder.to_tokens(tensor)
        print('Tokens from 2D tensor:', tokens)
        print('Extracted tokens metric:', main.Reporter.report('extracted_tokens'))
        self.assertEqual(tokens, [1, 2, 3, 4, 5])


if __name__ == '__main__':
    unittest.main()
