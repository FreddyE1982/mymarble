import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.tokenizer import StreamingTokenizer
from poted.decoder import StreamingDecoder


class TestStreamingDecoderLearning(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_roundtrip_with_mutations(self):
        tokenizer = StreamingTokenizer()
        decoder = StreamingDecoder(main.Reporter)
        stream1 = b'abab'
        stream2 = b'cdcd'
        tokens1 = tokenizer.tokenize(stream1)
        tokens2 = tokenizer.tokenize(stream2)
        tokens = tokens1[:-1] + tokens2[1:]
        decoded = decoder.decode(tokens)
        self.assertEqual(decoded, stream1 + stream2)
        self.assertEqual(main.Reporter.report('decoder_mutations'), 2)
        print('Combined tokens:', tokens)
        print('Decoded stream:', decoded)
        print('Decoder mutations metric:', main.Reporter.report('decoder_mutations'))


if __name__ == '__main__':
    unittest.main()
