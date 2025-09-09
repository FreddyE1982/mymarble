class StreamingTokenizer:
    def __init__(self, reporter=None, max_word_length=16, mode='volatile'):
        from .dictionary import DictionaryManager
        self._manager = DictionaryManager(reporter, max_word_length, mode)

    def encode(self, data):
        encoded = self._manager.encode(data)
        reporter = self._manager.reporter
        if reporter:
            steps = len(data)
            previous = reporter.report('tokenizer_steps') or 0
            reporter.report(
                'tokenizer_steps',
                'Estimated steps taken by tokenizer',
                previous + steps,
            )
            import sys
            memory = sys.getsizeof(encoded) + sys.getsizeof(self._manager._dict) + sys.getsizeof(self._manager._rev_dict)
            current = reporter.report('max_memory_bytes') or 0
            if memory > current:
                reporter.report('max_memory_bytes', 'Maximum memory usage in bytes', memory)
        return encoded

    def decode(self, tokens):
        decoded = self._manager.decode(tokens)
        reporter = self._manager.reporter
        if reporter:
            steps = len(tokens)
            previous = reporter.report('tokenizer_steps') or 0
            reporter.report(
                'tokenizer_steps',
                'Estimated steps taken by tokenizer',
                previous + steps,
            )
            import sys
            memory = sys.getsizeof(tokens) + sys.getsizeof(decoded) + sys.getsizeof(self._manager._dict) + sys.getsizeof(self._manager._rev_dict)
            current = reporter.report('max_memory_bytes') or 0
            if memory > current:
                reporter.report('max_memory_bytes', 'Maximum memory usage in bytes', memory)
        return decoded

    def tokenize(self, stream):
        from .control import ControlToken
        self._manager.reset()
        encoded = self.encode(stream)
        tokens = [
            int(ControlToken.BOS),
            int(ControlToken.RST),
            int(ControlToken.SYNC),
        ]
        tokens.extend(int(t) for t in encoded)
        tokens.append(int(ControlToken.EOS))
        if self._manager.reporter:
            self._manager.reporter.report(
                'control_tokens_added',
                'Number of control tokens added to stream',
                4,
            )
        return tokens

    def detokenize(self, tokens):
        from .core import Token
        from .control import ControlToken
        from .validator import ProtocolValidator
        ProtocolValidator.validate(tokens)
        payload = []
        for t in tokens:
            if t == int(ControlToken.RST):
                self._manager.reset()
                continue
            if t in (
                int(ControlToken.BOS),
                int(ControlToken.EOS),
                int(ControlToken.SYNC),
            ):
                continue
            payload.append(Token(t))
        decoded = self.decode(payload)
        return bytes(decoded)
