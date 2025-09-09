class StreamingTokenizer:
    def __init__(self, reporter=None, max_word_length=16):
        self._reporter = reporter
        self._max_word_length = max_word_length
        self._reset_state()

    def _reset_state(self):
        from .core import ByteAlphabet, Token
        self._alphabet = ByteAlphabet()
        self._dict = {}
        self._rev_dict = {}
        self._next = 0
        self._longest = 1
        for i in range(256):
            token = Token(self._next)
            self._dict[(i,)] = token
            self._rev_dict[int(token)] = (i,)
            self._next += 1
        if self._reporter:
            self._reporter.report(
                'dictionary_size',
                'Number of entries in tokenizer dictionary',
                self._next,
            )
            self._reporter.report(
                'longest_match',
                'Length of longest match during tokenization',
                self._longest,
            )

    def _add_sequence(self, seq):
        if len(seq) > self._max_word_length or seq in self._dict:
            return
        from .core import Token
        token = Token(self._next)
        self._dict[seq] = token
        self._rev_dict[int(token)] = seq
        self._next += 1
        if self._reporter:
            self._reporter.report('dictionary_size', value=self._next)

    def _update_longest(self, length):
        if length > self._longest:
            self._longest = length
            if self._reporter:
                self._reporter.report('longest_match', value=length)

    def encode(self, data):
        if isinstance(data, bytes):
            data = list(data)
        if not data:
            return []
        for b in data:
            if not self._alphabet.contains(b):
                raise ValueError('Byte out of alphabet')
        result = []
        w = (data[0],)
        for b in data[1:]:
            c = (b,)
            if len(w) < self._max_word_length and w + c in self._dict:
                w = w + c
            else:
                result.append(self._dict[w])
                self._update_longest(len(w))
                if len(w) < self._max_word_length:
                    self._add_sequence(w + c)
                w = c
        result.append(self._dict[w])
        self._update_longest(len(w))
        return result

    def decode(self, tokens):
        if not tokens:
            return []
        result = []
        prev = self._rev_dict.get(int(tokens[0]))
        if prev is None:
            raise KeyError('Unknown token')
        result.extend(prev)
        for t in tokens[1:]:
            seq = self._rev_dict.get(int(t))
            if seq is None:
                seq = prev + (prev[0],)
            result.extend(seq)
            self._add_sequence(prev + (seq[0],))
            prev = seq
        return result

    def tokenize(self, stream):
        from .control import ControlToken
        self._reset_state()
        encoded = self.encode(stream)
        tokens = [
            int(ControlToken.BOS),
            int(ControlToken.RST),
            int(ControlToken.SYNC),
        ]
        tokens.extend(int(t) for t in encoded)
        tokens.append(int(ControlToken.EOS))
        if self._reporter:
            self._reporter.report(
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
                self._reset_state()
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
