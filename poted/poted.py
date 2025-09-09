class PoTED:
    def __init__(
        self,
        Lw=0,
        Le=0,
        Lu=0,
        mode='instance',
        device='cpu',
        persistent=False,
        *,
        serializer=None,
        tokenizer=None,
        decoder=None,
        tensor_builder=None,
        reporter=None,
    ):
        reporter = self._instantiate_reporter(reporter)
        self._reporter = reporter
        self._mode = mode
        from .pipeline import JsonSerializer
        from .dictionary import DictionaryManager
        from .tokenizer import StreamingTokenizer
        from .decoder import StreamingDecoder
        from .tensor import TensorBuilder
        self._serializer = serializer if serializer is not None else JsonSerializer()
        self._dictionary = DictionaryManager(reporter, max_word_length=Lw, persistent=persistent)
        self._tokenizer = (
            tokenizer
            if tokenizer is not None
            else StreamingTokenizer(reporter=reporter, max_word_length=Lw, persistent=persistent)
        )
        self._tokenizer._manager = self._dictionary
        self._decoder = (
            decoder if decoder is not None else StreamingDecoder(reporter, persistent=persistent)
        )
        self._tensor_builder = (
            tensor_builder
            if tensor_builder is not None
            else TensorBuilder(Lw=Lw, Le=Le, Lu=Lu, device=device, reporter=reporter)
        )
        self._last_stream_hash = None
        self._last_dict_hash = None
    def _override(self, serializer=None, tokenizer=None, decoder=None, tensor_builder=None, reporter=None, **config):
        from contextlib import contextmanager

        @contextmanager
        def manager():
            original = {
                'serializer': self._serializer,
                'tokenizer': self._tokenizer,
                'decoder': self._decoder,
                'tensor_builder': self._tensor_builder,
                'reporter': self._reporter,
                'mode': self._mode,
                'device': getattr(self._tensor_builder, '_device', None),
                'Lw': getattr(self._tensor_builder, '_Lw', None),
                'Le': getattr(self._tensor_builder, '_Le', None),
                'Lu': getattr(self._tensor_builder, '_Lu', None),
            }
            try:
                if serializer is not None:
                    self._serializer = serializer
                if tokenizer is not None:
                    self._tokenizer = tokenizer
                    self._tokenizer._manager = self._dictionary
                if decoder is not None:
                    self._decoder = decoder
                if tensor_builder is not None:
                    self._tensor_builder = tensor_builder
                if reporter is not None:
                    self._reporter = reporter
                    self._tokenizer._manager._reporter = reporter
                    if hasattr(self._decoder, '_reporter'):
                        self._decoder._reporter = reporter
                    if hasattr(self._tensor_builder, '_reporter'):
                        self._tensor_builder._reporter = reporter
                if 'mode' in config:
                    self._mode = config['mode']
                if 'device' in config:
                    import torch
                    self._tensor_builder._device = torch.device(config['device'])
                if 'Lw' in config:
                    self._tensor_builder._Lw = config['Lw']
                if 'Le' in config:
                    self._tensor_builder._Le = config['Le']
                if 'Lu' in config:
                    self._tensor_builder._Lu = config['Lu']
                yield
            finally:
                self._serializer = original['serializer']
                self._tokenizer = original['tokenizer']
                self._decoder = original['decoder']
                self._tensor_builder = original['tensor_builder']
                self._reporter = original['reporter']
                self._mode = original['mode']
                if original['device'] is not None:
                    self._tensor_builder._device = original['device']
                if original['Lw'] is not None:
                    self._tensor_builder._Lw = original['Lw']
                if original['Le'] is not None:
                    self._tensor_builder._Le = original['Le']
                if original['Lu'] is not None:
                    self._tensor_builder._Lu = original['Lu']
                self._tokenizer._manager._reporter = original['reporter']
                if hasattr(self._decoder, '_reporter'):
                    self._decoder._reporter = original['reporter']
                if hasattr(self._tensor_builder, '_reporter'):
                    self._tensor_builder._reporter = original['reporter']
        return manager()
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
        if self._mode == 'portable':
            payload = {
                'dictionary': self._dictionary.export(),
                'payload': obj,
            }
            stream = self._serializer.serialize(payload)
        else:
            stream = self._serializer.serialize(obj)
        tokens = self._tokenizer.tokenize(stream)
        tensor = self._tensor_builder.to_tensor(tokens)
        total = len(tokens)
        ratio = len(tokens) / len(stream) if stream else 0
        self._reporter.report('total_tokens', 'Total number of tokens', total)
        self._reporter.report('compression_ratio', 'Token count to byte length ratio', ratio)
        self._reporter.report(
            'tensor_shape',
            'Shape of tensor produced by TensorBuilder',
            list(tensor.shape),
        )
        count = self._reporter.report('encode_calls') or 0
        self._reporter.report('encode_calls', 'Number of encode operations', count + 1)
        return tensor
    def decode(self, tensor, *, serializer=None, tokenizer=None, decoder=None, tensor_builder=None, reporter=None, **config):
        with self._override(serializer, tokenizer, decoder, tensor_builder, reporter, **config):
            try:
                tokens = self._tensor_builder.to_tokens(tensor)
                from .validator import ProtocolValidator
                ProtocolValidator.validate(tokens)
                from .control import ControlToken
                resets = sum(1 for t in tokens if t == int(ControlToken.RST))
                self._reporter.report(
                    'sync_resets',
                    'Number of decoder synchronisation resets',
                    max(0, resets - 1),
                )
                from .integrity import IntegrityChecker
                checker = IntegrityChecker(self._reporter)
                stream_hash = checker.hash_stream(tokens)
                if self._last_stream_hash is not None and stream_hash != self._last_stream_hash:
                    from .errors import SyncError
                    raise SyncError('Stream hash mismatch')
                stream = self._decoder.decode(tokens)
                obj = self._serializer.deserialize(stream)
                if self._mode == 'portable' and isinstance(obj, dict) and 'payload' in obj:
                    obj = obj['payload']
                if self._last_dict_hash is not None:
                    from types import SimpleNamespace
                    expected = getattr(self._tokenizer, "_manager", None)
                    limit = getattr(expected, "_next", None)
                    mapping = {
                        seq: token
                        for token, seq in self._decoder._rev_dict.items()
                        if limit is None or token < limit
                    }
                    dict_hash = checker.hash_dictionary(SimpleNamespace(_dict=mapping))
                    if dict_hash != self._last_dict_hash:
                        from .errors import DictionaryMismatch
                        raise DictionaryMismatch('Dictionary hash mismatch')
            except Exception:
                count = self._reporter.report('roundtrip_failures') or 0
                self._reporter.report(
                    'roundtrip_failures',
                    'Number of failed roundtrip comparisons',
                    count + 1,
                )
                raise
            count = self._reporter.report('decode_calls') or 0
            self._reporter.report('decode_calls', 'Number of decode operations', count + 1)
            return obj
    def __call__(self, obj, *, serializer=None, tokenizer=None, decoder=None, tensor_builder=None, reporter=None, **config):
        with self._override(serializer, tokenizer, decoder, tensor_builder, reporter, **config):
            stream = self._serializer.serialize(obj)
            from .control import ControlToken
            self._tokenizer._manager.reset()
            payload = self._tokenizer.encode(stream)
            tokens = [int(ControlToken.BOS)]
            if self._dictionary._mode != 'persistent':
                tokens.append(int(ControlToken.RST))
            tokens.append(int(ControlToken.SYNC))
            tokens.extend(int(t) for t in payload)
            tokens.append(int(ControlToken.EOS))
            from .integrity import IntegrityChecker
            checker = IntegrityChecker(self._reporter)
            self._last_stream_hash = checker.hash_stream(tokens)
            self._last_dict_hash = checker.hash_dictionary(self._tokenizer._manager)
            tensor = self._tensor_builder.to_tensor(tokens)
            total = len(tokens)
            ratio = len(tokens) / len(stream) if stream else 0
            self._reporter.report('total_tokens', 'Total number of tokens', total)
            self._reporter.report('compression_ratio', 'Token count to byte length ratio', ratio)
            self._reporter.report(
                'tensor_shape',
                'Shape of tensor produced by TensorBuilder',
                list(tensor.shape),
            )
            count = self._reporter.report('encode_calls') or 0
            self._reporter.report('encode_calls', 'Number of encode operations', count + 1)
            if self._dictionary._mode != 'persistent':
                self._dictionary.reset()
        return tensor
