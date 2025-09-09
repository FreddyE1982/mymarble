class TensorBuilder:
    PAD = -5

    def __init__(self, Lw=0, Le=0, Lu=0, device='cpu', reporter=None):
        import torch
        self._Lw = Lw
        self._Le = Le
        self._Lu = Lu
        self._device = torch.device(device)
        self._reporter = reporter

    def from_tokens(self, tokens, device=None):
        import torch
        sequences = tokens if tokens and isinstance(tokens[0], list) else [tokens]
        L = self._Lw + self._Le + self._Lu
        if L <= 0:
            L = max((len(seq) for seq in sequences), default=0)
        padded = []
        for seq in sequences:
            if len(seq) < L:
                seq = seq + [self.PAD] * (L - len(seq))
            else:
                seq = seq[:L]
            padded.append(seq)
        target_device = torch.device(device) if device is not None else self._device
        tensor = torch.tensor(padded, dtype=torch.int64, device=target_device)
        if self._reporter:
            self._reporter.report(
                'tensor_shape',
                'Shape of tensor produced by TensorBuilder',
                list(tensor.shape),
            )
        return tensor

    def to_tensor(self, tokens, device=None):
        return self.from_tokens(tokens, device=device)

    def to_tokens(self, tensor):
        tokens = []
        for row in tensor.tolist():
            for x in row:
                if x != self.PAD:
                    tokens.append(int(x))
        return tokens
