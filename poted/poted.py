class PoTED:
    def __init__(self, *, serializer=None, tokenizer=None, decoder=None, tensor_builder=None, reporter=None, **config):
        reporter = self._instantiate_reporter(reporter)
        self._reporter = reporter
        dict_mode = 'persistent' if config.get('persistent') else config.get('mode', 'volatile')
        Lw = config.get('Lw', 0)
        Le = config.get('Le', 0)
        Lu = config.get('Lu', 0)
        device = config.get('device', 'cpu')
        from .pipeline import JsonSerializer
        from .dictionary import DictionaryManager
        from .tokenizer import StreamingTokenizer
        from .decoder import StreamingDecoder
        from .tensor import TensorBuilder
        self._serializer = serializer if serializer is not None else JsonSerializer()
        self._dictionary = DictionaryManager(reporter, mode=dict_mode)
        self._tokenizer = tokenizer if tokenizer is not None else StreamingTokenizer(reporter=reporter, mode=dict_mode)
        self._tokenizer._manager = self._dictionary
        self._decoder = decoder if decoder is not None else StreamingDecoder(reporter)
        self._tensor_builder = (
            tensor_builder
            if tensor_builder is not None
            else TensorBuilder(Lw=Lw, Le=Le, Lu=Lu, device=device, reporter=reporter)
        )
    def _instantiate_reporter(self, reporter):
        if reporter is None:
            reporter = __import__('main').Reporter
        if isinstance(reporter, type):
            reporter = reporter()
        return reporter
    @property
    def reporter(self):
        return self._reporter
    @property
    def serializer(self):
        return self._serializer
    @property
    def tokenizer(self):
        return self._tokenizer
    @property
    def decoder(self):
        return self._decoder
    @property
    def tensor_builder(self):
        return self._tensor_builder
    def encode(self, obj):
        stream = self._serializer.serialize(obj)
        tokens = self._tokenizer.tokenize(stream)
        tensor = self._tensor_builder.to_tensor(tokens)
        count = self._reporter.report('encode_calls') or 0
        self._reporter.report('encode_calls', 'Number of encode operations', count + 1)
        return tensor
    def decode(self, tensor):
        tokens = self._tensor_builder.to_tokens(tensor)
        stream = self._decoder.decode(tokens)
        obj = self._serializer.deserialize(stream)
        count = self._reporter.report('decode_calls') or 0
        self._reporter.report('decode_calls', 'Number of decode operations', count + 1)
        return obj
