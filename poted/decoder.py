class StreamingDecoder:
    def __init__(self, reporter=None):
        self._reporter = reporter
        self._reset_state()

    def _reset_state(self):
        from .core import Token
        self._rev_dict = {}
        self._next = 0
        for i in range(256):
            token = Token(self._next)
            self._rev_dict[int(token)] = (i,)
            self._next += 1

    def _add_sequence(self, seq):
        from .core import Token
        token = Token(self._next)
        self._rev_dict[int(token)] = seq
        self._next += 1

    def decode(self, tokens):
        from .core import Token
        from .control import ControlToken
        from .validator import ProtocolValidator
        ProtocolValidator.validate(tokens)
        result = []
        prev = None
        for t in tokens:
            if t == int(ControlToken.RST):
                self._reset_state()
                if self._reporter:
                    count = self._reporter.report('decoder_mutations') or 0
                    self._reporter.report(
                        'decoder_mutations',
                        'Number of decoder state resets',
                        count + 1,
                    )
                prev = None
                continue
            if t in (
                int(ControlToken.BOS),
                int(ControlToken.EOS),
                int(ControlToken.SYNC),
            ):
                continue
            token = Token(t)
            seq = self._rev_dict.get(int(token))
            if seq is None:
                if prev is None:
                    raise KeyError('Unknown token')
                seq = prev + (prev[0],)
            result.extend(seq)
            if prev is not None:
                self._add_sequence(prev + (seq[0],))
            prev = seq
        return bytes(result)
