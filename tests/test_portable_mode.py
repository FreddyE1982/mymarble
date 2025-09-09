import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.pipeline import JsonSerializer, TensorBuilder, PoTEDPipeline
from poted.tokenizer import StreamingTokenizer


class TestPortableMode(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}
        self.serializer = JsonSerializer()
        self.tokenizer = StreamingTokenizer()
        self.builder = TensorBuilder()
        self.pipeline = PoTEDPipeline(
            self.serializer, self.tokenizer, self.builder, mode='portable'
        )

    def test_dictionary_included(self):
        obj = {"numbers": [1, 2, 3]}
        tensor = self.pipeline.encode(obj)
        tokens = self.builder.to_tokens(tensor)
        stream = self.tokenizer.detokenize(tokens)
        wrapped = self.serializer.deserialize(stream)
        self.assertIn('dictionary', wrapped)
        self.assertIn('payload', wrapped)
        self.assertEqual(wrapped['payload'], obj)
        result = self.pipeline.decode(tensor)
        self.assertEqual(result, obj)
        print('Dictionary entries:', len(wrapped['dictionary']))


if __name__ == '__main__':
    unittest.main()
