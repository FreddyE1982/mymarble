class ByteAlphabet:
    def __init__(self):
        self._symbols = []
        for i in range(256):
            self._symbols.append(i)

    def contains(self, byte):
        return isinstance(byte, int) and 0 <= byte < 256


class Token(int):
    def __new__(cls, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError('Token value must be non-negative integer')
        return int.__new__(cls, value)

    def __repr__(self):
        return 'Token(%d)' % int(self)


class Dictionary:
    def __init__(self, reporter):
        self._b2t = {}
        self._t2b = {}
        self._next = 0
        self._reporter = reporter
        self._update_metrics()

    def _update_metrics(self):
        if self._reporter:
            self._reporter.report('dictionary_size', 'Number of entries in tokenizer dictionary', self.size)

    @property
    def size(self):
        return len(self._b2t)

    def add(self, byte):
        token = self._b2t.get(byte)
        if token is None:
            token = Token(self._next)
            self._b2t[byte] = token
            self._t2b[int(token)] = byte
            self._next += 1
            self._update_metrics()
        return token

    def token(self, byte):
        return self._b2t.get(byte)

    def byte(self, token):
        return self._t2b.get(int(token))


class TokenizerState:
    def __init__(self, reporter):
        self._alphabet = ByteAlphabet()
        self._dictionary = Dictionary(reporter)

    def encode_byte(self, byte):
        if not self._alphabet.contains(byte):
            raise ValueError('Byte out of alphabet')
        return self._dictionary.add(byte)

    def decode_token(self, token):
        byte = self._dictionary.byte(token)
        if byte is None:
            raise KeyError('Unknown token')
        return byte

    @property
    def dictionary(self):
        return self._dictionary
