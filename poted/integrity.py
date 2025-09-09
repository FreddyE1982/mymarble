class IntegrityChecker:
    def __init__(self, reporter=None, segment_size=64):
        """Compute cryptographic hashes for token streams and dictionaries.

        Args:
            reporter: Optional reporter used for metric collection.
            segment_size: Number of tokens per segment when hashing streams.
        """
        self._reporter = reporter
        self._segment_size = segment_size

    def _hash_bytes(self, data):
        """Return SHA256 hexadecimal digest for *data* bytes."""
        import hashlib

        hasher = hashlib.sha256()
        hasher.update(data)
        return hasher.hexdigest()

    def _token_bytes(self, token):
        """Encode a token as 4-byte big-endian representation."""
        return int(token).to_bytes(4, 'big', signed=False)

    def hash_stream(self, stream):
        """Hash a sequence of tokens or bytes in fixed-size segments.

        Segment hashes and the resulting stream hash are reported under
        the metric names ``segment_hashes`` and ``stream_hash`` when a
        reporter is provided.
        """
        if stream is None:
            tokens = []
        elif isinstance(stream, bytes):
            tokens = list(stream)
        else:
            tokens = list(stream)

        segment_hashes = []
        for i in range(0, len(tokens), self._segment_size):
            segment = tokens[i : i + self._segment_size]
            seg_bytes = b"".join(self._token_bytes(t) for t in segment)
            segment_hashes.append(self._hash_bytes(seg_bytes))

        stream_input = "".join(segment_hashes).encode("utf-8")
        stream_hash = self._hash_bytes(stream_input)

        if self._reporter:
            self._reporter.report(
                "segment_hashes",
                "SHA256 hashes of individual stream segments",
                segment_hashes,
            )
            self._reporter.report(
                "stream_hash", "SHA256 hash of the entire stream", stream_hash
            )

        return stream_hash

    def hash_dictionary(self, dictionary):
        """Hash dictionary mappings in a deterministic manner."""
        items = []
        if hasattr(dictionary, "_dict"):
            for seq, token in dictionary._dict.items():
                seq_bytes = bytes(seq)
                token_bytes = self._token_bytes(token)
                items.append((seq_bytes, token_bytes))
        elif hasattr(dictionary, "_b2t"):
            for byte, token in dictionary._b2t.items():
                seq_bytes = bytes([byte])
                token_bytes = self._token_bytes(token)
                items.append((seq_bytes, token_bytes))
        else:
            raise TypeError("Unsupported dictionary type")

        items.sort(key=lambda x: x[0])
        segment_hashes = []
        for seq_bytes, token_bytes in items:
            segment_hashes.append(self._hash_bytes(seq_bytes + token_bytes))

        digest = self._hash_bytes("".join(segment_hashes).encode("utf-8"))

        if self._reporter:
            self._reporter.report(
                "segment_hashes",
                "SHA256 hashes of dictionary segments",
                segment_hashes,
            )
            self._reporter.report(
                "stream_hash", "SHA256 hash of the entire dictionary", digest
            )

        return digest
