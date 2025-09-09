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
    PAD = -5

    def __init__(self, Lw=0, Le=0, Lu=0, reporter=None):
        self._Lw = Lw
        self._Le = Le
        self._Lu = Lu
        self._reporter = reporter

    def to_tensor(self, tokens):
        import torch
        sequences = tokens if tokens and isinstance(tokens[0], list) else [tokens]
        L = self._Lw + self._Le + self._Lu
        if L <= 0:
            L = max((len(seq) for seq in sequences), default=0)
        padded = []
        for seq in sequences:
            if len(seq) < L:
                seq = seq + [self.PAD] * (L - len(seq))
            else:
                seq = seq[:L]
            padded.append(seq)
        tensor = torch.tensor(padded, dtype=torch.int64)
        if self._reporter:
            self._reporter.report(
                'tensor_shape',
                'Shape of tensor produced by TensorBuilder',
                list(tensor.shape),
            )
        return tensor

    def to_tokens(self, tensor):
        tokens = []
        for row in tensor.tolist():
            for x in row:
                if x != self.PAD:
                    tokens.append(int(x))
        return tokens


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
