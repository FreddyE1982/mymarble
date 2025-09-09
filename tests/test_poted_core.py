import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.core import TokenizerState


class TestTokenizerState(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_encode_decode_bijection(self):
        state = TokenizerState(main.Reporter)
        for b in range(256):
            token = state.encode_byte(b)
            decoded = state.decode_token(token)
            self.assertEqual(decoded, b)
        size = main.Reporter.report('dictionary_size')
        print('Dictionary size after encoding:', size)
        self.assertEqual(size, 256)

    def test_decode_encode_bijection(self):
        state = TokenizerState(main.Reporter)
        tokens = [state.encode_byte(b) for b in range(256)]
        for token in tokens:
            byte = state.decode_token(token)
            token2 = state.encode_byte(byte)
            self.assertEqual(int(token2), int(token))
        unique_tokens = len({int(t) for t in tokens})
        print('Unique tokens:', unique_tokens)
        self.assertEqual(unique_tokens, 256)


if __name__ == '__main__':
    unittest.main()
