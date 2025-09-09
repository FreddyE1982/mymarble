class SyncError(Exception):
    """Raised when a synchronization issue occurs in the token stream."""
    def __init__(self, message="Token stream out of sync"):
        super().__init__(message)


class DictionaryMismatch(SyncError):
    """Raised when decoder and encoder dictionaries become inconsistent."""
    def __init__(self, message="Dictionary mismatch detected"):
        super().__init__(message)
