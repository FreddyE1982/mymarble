import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.pipeline import JsonSerializer, TensorBuilder, PoTEDPipeline
from poted.tokenizer import StreamingTokenizer


class TestInstanceMode(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}
        self.serializer = JsonSerializer()
        self.tokenizer = StreamingTokenizer()
        self.builder = TensorBuilder()
        self.pipeline = PoTEDPipeline(
            self.serializer, self.tokenizer, self.builder, mode='instance'
        )

    def test_no_dictionary_included(self):
        obj = {"values": [4, 5, 6]}
        tensor = self.pipeline.encode(obj)
        tokens = self.builder.to_tokens(tensor)
        stream = self.tokenizer.detokenize(tokens)
        decoded = self.serializer.deserialize(stream)
        self.assertEqual(decoded, obj)
        self.assertNotIn('dictionary', decoded)
        result = self.pipeline.decode(tensor)
        self.assertEqual(result, obj)
        print('Token count:', len(tokens))


if __name__ == '__main__':
    unittest.main()
