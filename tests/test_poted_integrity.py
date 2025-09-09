import sys
import pathlib
import unittest

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.poted import PoTED
from poted.errors import SyncError


class TestPoTEDIntegrity(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_stream_hash_mismatch_raises(self):
        engine = PoTED(reporter=main.Reporter)
        tensor = engine({'msg': 'hi'})
        tokens = engine.tensor_builder.to_tokens(tensor)
        tokens[3] = (tokens[3] + 1) % 256
        tampered = engine.tensor_builder.to_tensor(tokens)
        with self.assertRaises(SyncError):
            engine.decode(tampered)
        segment_hashes = main.Reporter.report('segment_hashes')
        stream_hash = main.Reporter.report('stream_hash')
        sync_resets = main.Reporter.report('sync_resets')
        print('Hashes:', segment_hashes, stream_hash)
        print('Sync resets:', sync_resets)
        self.assertIsNotNone(segment_hashes)
        self.assertIsNotNone(stream_hash)
        self.assertIsNotNone(sync_resets)


if __name__ == '__main__':
    unittest.main()
