import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.integrity import IntegrityChecker
from poted.dictionary import DictionaryManager


class TestIntegrity(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}
        self.checker = IntegrityChecker(main.Reporter, segment_size=4)

    def test_hash_stream_and_corruption(self):
        tokens = [1, 2, 3, 4, 5, 6]
        original = self.checker.hash_stream(tokens)
        segments = main.Reporter.report('segment_hashes')
        stream = main.Reporter.report('stream_hash')
        print('Segment hashes:', segments)
        print('Stream hash:', stream)
        self.assertEqual(original, stream)
        self.assertEqual(len(segments), 2)
        corrupted = list(tokens)
        corrupted[3] = 9999
        altered = self.checker.hash_stream(corrupted)
        print('Corrupted stream hash:', altered)
        self.assertNotEqual(original, altered)

    def test_hash_dictionary_corruption(self):
        manager = DictionaryManager(main.Reporter, max_word_length=4)
        manager.encode(b'abracadabra')
        first = self.checker.hash_dictionary(manager)
        print('Dictionary hash before:', first)
        del manager._dict[(97,)]
        second = self.checker.hash_dictionary(manager)
        print('Dictionary hash after:', second)
        self.assertNotEqual(first, second)


if __name__ == '__main__':
    unittest.main()
