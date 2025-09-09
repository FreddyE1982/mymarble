import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.decoder import StreamingDecoder
from poted.errors import SyncError, DictionaryMismatch
from poted.control import ControlToken


class TestErrorCases(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_dictionary_mismatch_triggers_reset(self):
        decoder = StreamingDecoder(reporter=main.Reporter)
        tokens = [
            int(ControlToken.BOS),
            int(ControlToken.RST),
            int(ControlToken.SYNC),
            999,
            int(ControlToken.EOS),
        ]
        with self.assertRaises(DictionaryMismatch):
            decoder.decode(tokens)
        resets = main.Reporter.report('sync_resets')
        print('Sync resets:', resets)
        self.assertEqual(resets, 1)

    def test_dictionary_mismatch_out_of_range(self):
        decoder = StreamingDecoder(reporter=main.Reporter)
        tokens = [
            int(ControlToken.BOS),
            int(ControlToken.RST),
            int(ControlToken.SYNC),
            65,
            999,
            int(ControlToken.EOS),
        ]
        with self.assertRaises(DictionaryMismatch):
            decoder.decode(tokens)
        resets = main.Reporter.report('sync_resets')
        print('Sync resets:', resets)
        self.assertEqual(resets, 1)

    def test_exception_hierarchy(self):
        self.assertTrue(issubclass(DictionaryMismatch, SyncError))


if __name__ == '__main__':
    unittest.main()
