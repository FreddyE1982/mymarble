import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.tokenizer import StreamingTokenizer


class TestDictionaryPersistence(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_volatile_reset(self):
        tokenizer = StreamingTokenizer(main.Reporter, max_word_length=8, mode='volatile')
        stream = [65, 66, 65, 66]
        tokenizer.tokenize(stream)
        size_first = main.Reporter.report('dictionary_size')
        tokens = tokenizer.tokenize(stream)
        tokenizer.detokenize(tokens)
        size_second = main.Reporter.report('dictionary_size')
        print('Volatile size after first:', size_first)
        print('Volatile size after second:', size_second)
        self.assertEqual(size_second, size_first)

    def test_persistent_keeps_dictionary(self):
        tokenizer = StreamingTokenizer(main.Reporter, max_word_length=8, mode='persistent')
        stream = [65, 66, 65, 66]
        tokenizer.tokenize(stream)
        size_before = main.Reporter.report('dictionary_size')
        tokens = tokenizer.tokenize(stream)
        tokenizer.detokenize(tokens)
        size_after = main.Reporter.report('dictionary_size')
        sessions = main.Reporter.report('persistent_sessions')
        print('Persistent size before:', size_before)
        print('Persistent size after:', size_after)
        print('Persistent sessions metric:', sessions)
        self.assertTrue(size_after > size_before)
        self.assertEqual(sessions, 1)


if __name__ == '__main__':
    unittest.main()
