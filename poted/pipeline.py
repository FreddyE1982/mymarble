from poted.tensor import TensorBuilder


class JsonSerializer:
    def serialize(self, obj):
        import json
        import numpy as np
        import torch

        def default(o):
            if isinstance(o, np.ndarray):
                return {"__ndarray__": o.tolist(), "dtype": str(o.dtype)}
            if isinstance(o, torch.Tensor):
                return {
                    "__torch_tensor__": o.tolist(),
                    "dtype": str(o.dtype),
                    "device": str(o.device),
                    "requires_grad": o.requires_grad,
                }
            raise TypeError(
                f"Object of type {type(o).__name__} is not JSON serializable"
            )

        data = json.dumps(obj, separators=(",", ":"), sort_keys=True, default=default)
        return data.encode("utf-8")

    def deserialize(self, stream):
        import json
        import numpy as np
        import torch

        def object_hook(d):
            if "__ndarray__" in d:
                return np.array(d["__ndarray__"], dtype=d.get("dtype"))
            if "__torch_tensor__" in d:
                dtype_name = d.get("dtype", "float")
                dtype = getattr(torch, dtype_name.split(".")[-1])
                device = torch.device(d.get("device", "cpu"))
                requires_grad = d.get("requires_grad", False)
                return torch.tensor(
                    d["__torch_tensor__"],
                    dtype=dtype,
                    device=device,
                    requires_grad=requires_grad,
                )
            return d

        text = stream.decode("utf-8")
        return json.loads(text, object_hook=object_hook)


class PoTEDPipeline:
    def __init__(
        self, serializer, tokenizer, tensor_builder, reporter=None, mode="instance"
    ):
        if mode not in ("portable", "instance"):
            raise ValueError("mode must be 'portable' or 'instance'")
        self._serializer = serializer
        self._tokenizer = tokenizer
        self._tensor_builder = tensor_builder
        self._reporter = reporter
        self._mode = mode

    def encode(self, obj):
        if self._mode == "portable":
            dictionary = self._tokenizer._manager.export()
            payload = {"dictionary": dictionary, "payload": obj}
            stream = self._serializer.serialize(payload)
        else:
            stream = self._serializer.serialize(obj)
        tokens = self._tokenizer.tokenize(stream)
        tensor = self._tensor_builder.to_tensor(tokens)
        if self._reporter:
            total = len(tokens)
            self._reporter.report("total_tokens", "Total number of tokens", total)
            ratio = len(tokens) / len(stream) if stream else 0
            self._reporter.report(
                "compression_ratio", "Token count to byte length ratio", ratio
            )
        return tensor

    def decode(self, tensor):
        tokens = self._tensor_builder.to_tokens(tensor)
        stream = self._tokenizer.detokenize(tokens)
        obj = self._serializer.deserialize(stream)
        if self._mode == "portable" and isinstance(obj, dict) and "payload" in obj:
            return obj["payload"]
        return obj
