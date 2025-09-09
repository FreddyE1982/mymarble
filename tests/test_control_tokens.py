import sys
import pathlib
import unittest

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.tokenizer import StreamingTokenizer
from poted.control import ControlToken
from poted.validator import ProtocolValidator


class TestControlTokens(unittest.TestCase):
    def test_tokenize_inserts_control_tokens(self):
        stream = b'friend'
        tokenizer = StreamingTokenizer(main.Reporter)
        tokens = tokenizer.tokenize(stream)
        print('Tokenized with control tokens:', tokens)
        self.assertEqual(tokens[0], int(ControlToken.BOS))
        self.assertEqual(tokens[1], int(ControlToken.RST))
        self.assertEqual(tokens[2], int(ControlToken.SYNC))
        self.assertEqual(tokens[-1], int(ControlToken.EOS))
        payload = tokens[3:-1]
        self.assertTrue(all(t >= 0 for t in payload))

    def test_detokenize_roundtrip(self):
        stream = b'buddyo'
        tokenizer = StreamingTokenizer(main.Reporter)
        tokens = tokenizer.tokenize(stream)
        decoded = tokenizer.detokenize(tokens)
        print('Decoded stream:', decoded)
        self.assertEqual(decoded, stream)

    def test_validator(self):
        stream = b'pal'
        tokenizer = StreamingTokenizer(main.Reporter)
        tokens = tokenizer.tokenize(stream)
        print('Validating tokens:', tokens)
        self.assertTrue(ProtocolValidator.validate(tokens))
        with self.assertRaises(ValueError):
            ProtocolValidator.validate(tokens[1:])


if __name__ == '__main__':
    unittest.main()
