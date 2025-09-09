import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.tokenizer import StreamingTokenizer


class TestStreamingTokenizer(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_roundtrip(self):
        tokenizer = StreamingTokenizer(main.Reporter, max_word_length=8)
        stream = [65, 66, 65, 66, 65, 66, 65, 66]
        tokens = tokenizer.encode(stream)
        decoded = tokenizer.decode(tokens)
        print('Encoded tokens:', tokens)
        print('Decoded stream:', decoded)
        print('Metrics:', main.Reporter._metrics)
        self.assertEqual(decoded, stream)

    def test_longest_match_metric(self):
        tokenizer = StreamingTokenizer(main.Reporter, max_word_length=10)
        stream = [65] * 6
        tokenizer.encode(stream)
        longest = main.Reporter.report('longest_match')
        print('Longest match metric:', longest)
        self.assertEqual(longest, 3)

    def test_word_length_limit(self):
        tokenizer = StreamingTokenizer(main.Reporter, max_word_length=2)
        stream = [65, 65, 65, 65]
        tokenizer.encode(stream)
        size = main.Reporter.report('dictionary_size')
        longest = main.Reporter.report('longest_match')
        print('Dictionary size:', size)
        print('Longest match:', longest)
        self.assertEqual(size, 257)
        self.assertEqual(longest, 2)

    def test_invalid_token_raises_keyerror(self):
        tokenizer = StreamingTokenizer(main.Reporter)
        stream = [65, 66, 67]
        tokens = tokenizer.encode(stream)
        tokens.append(9999)
        print('Corrupted tokens:', tokens)
        with self.assertRaises(KeyError):
            tokenizer.decode(tokens)


if __name__ == '__main__':
    unittest.main()
