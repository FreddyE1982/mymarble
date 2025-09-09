import unittest
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import main
from poted.tokenizer import StreamingTokenizer
from poted.pipeline import JsonSerializer, TensorBuilder, PoTEDPipeline


class TestPoTEDPipelineRoundtrip(unittest.TestCase):
    def setUp(self):
        main.Reporter._metrics = {}
        self.serializer = JsonSerializer()
        self.tokenizer = StreamingTokenizer(main.Reporter)
        self.builder = TensorBuilder()
        self.pipeline = PoTEDPipeline(
            self.serializer, self.tokenizer, self.builder, main.Reporter
        )

    def test_roundtrip(self):
        obj = {"numbers": [1, 2, 3], "nested": {"flag": True}}
        tensor = self.pipeline.encode(obj)
        result = self.pipeline.decode(tensor)
        self.assertEqual(result, obj)
        stream = self.serializer.serialize(obj)
        tokens = StreamingTokenizer().tokenize(stream)
        expected_ratio = len(tokens) / len(stream)
        self.assertEqual(main.Reporter.report('total_tokens'), len(tokens))
        self.assertAlmostEqual(
            main.Reporter.report('compression_ratio'), expected_ratio
        )
        print('Total tokens:', main.Reporter.report('total_tokens'))
        print('Compression ratio:', main.Reporter.report('compression_ratio'))


if __name__ == '__main__':
    unittest.main()
