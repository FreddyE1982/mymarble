class StreamingDecoder:
    def __init__(self, reporter=None, persistent=False):
        self._reporter = reporter
        self._mode = 'persistent' if persistent else 'volatile'
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
        if self._reporter:
            count = self._reporter.report('decoder_mutations') or 0
            self._reporter.report(
                'decoder_mutations',
                'Number of decoder dictionary mutations',
                count + 1,
            )

    def decode(self, tokens):
        from .core import Token
        from .control import ControlToken
        from .validator import ProtocolValidator
        ProtocolValidator.validate(tokens)
        if self._mode == 'volatile':
            self._reset_state()
        result = []
        prev = None
        for t in tokens:
            if t == int(ControlToken.RST):
                self._reset_state()
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
                if prev is None or int(token) != self._next:
                    self._reset_state()
                    if self._reporter:
                        count = self._reporter.report('sync_resets') or 0
                        self._reporter.report(
                            'sync_resets',
                            'Number of decoder synchronisation resets',
                            count + 1,
                        )
                    from .errors import DictionaryMismatch
                    raise DictionaryMismatch('Unknown token')
                seq = prev + (prev[0],)
            result.extend(seq)
            if prev is not None:
                self._add_sequence(prev + (seq[0],))
            prev = seq
        output = bytes(result)
        if self._reporter:
            steps = len(tokens)
            previous = self._reporter.report('decoder_steps') or 0
            self._reporter.report(
                'decoder_steps',
                'Estimated steps taken by decoder',
                previous + steps,
            )
            import sys
            memory = sys.getsizeof(tokens) + sys.getsizeof(output) + sys.getsizeof(self._rev_dict)
            current = self._reporter.report('max_memory_bytes') or 0
            if memory > current:
                self._reporter.report('max_memory_bytes', 'Maximum memory usage in bytes', memory)
        if self._mode == 'volatile':
            self._reset_state()
        return output
