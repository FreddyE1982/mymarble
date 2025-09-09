import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.tokenizer import StreamingTokenizer
from poted.decoder import StreamingDecoder


class TestComplexityReporting(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}

    def test_complexity_metrics(self):
        tokenizer = StreamingTokenizer(main.Reporter)
        decoder = StreamingDecoder(main.Reporter)
        data = b"hello world"
        tokens = tokenizer.tokenize(data)
        decoded = decoder.decode(tokens)
        self.assertEqual(decoded, data)
        tok_steps = main.Reporter.report('tokenizer_steps')
        dec_steps = main.Reporter.report('decoder_steps')
        max_mem = main.Reporter.report('max_memory_bytes')
        print('Tokenizer steps:', tok_steps)
        print('Decoder steps:', dec_steps)
        print('Max memory bytes:', max_mem)
        self.assertGreaterEqual(tok_steps, len(data))
        self.assertGreaterEqual(dec_steps, len(tokens))
        self.assertGreater(max_mem, 0)


if __name__ == '__main__':
    unittest.main()
