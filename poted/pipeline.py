class JsonSerializer:
    def serialize(self, obj):
        import json
        data = json.dumps(obj, separators=(",", ":"), sort_keys=True)
        return data.encode('utf-8')

    def deserialize(self, stream):
        import json
        text = stream.decode('utf-8')
        return json.loads(text)


class TensorBuilder:
    def __init__(self, reporter=None):
        self._reporter = reporter

    def to_tensor(self, tokens):
        import torch
        tensor = torch.tensor(tokens, dtype=torch.int64)
        return tensor

    def to_tokens(self, tensor):
        return [int(x) for x in tensor.tolist()]


class PoTEDPipeline:
    def __init__(self, serializer, tokenizer, tensor_builder, reporter=None):
        self._serializer = serializer
        self._tokenizer = tokenizer
        self._tensor_builder = tensor_builder
        self._reporter = reporter

    def encode(self, obj):
        stream = self._serializer.serialize(obj)
        tokens = self._tokenizer.tokenize(stream)
        tensor = self._tensor_builder.to_tensor(tokens)
        if self._reporter:
            total = len(tokens)
            self._reporter.report('total_tokens', 'Total number of tokens', total)
            ratio = len(tokens) / len(stream) if stream else 0
            self._reporter.report('compression_ratio', 'Token count to byte length ratio', ratio)
        return tensor

    def decode(self, tensor):
        tokens = self._tensor_builder.to_tokens(tensor)
        stream = self._tokenizer.detokenize(tokens)
        return self._serializer.deserialize(stream)
