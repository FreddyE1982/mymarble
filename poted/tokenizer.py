class StreamingTokenizer:
    def __init__(self, reporter=None, max_word_length=16, mode='volatile'):
        from .dictionary import DictionaryManager
        self._manager = DictionaryManager(reporter, max_word_length, mode)

    def encode(self, data):
        return self._manager.encode(data)

    def decode(self, tokens):
        return self._manager.decode(tokens)

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
