# mymarble

## Quickstart with PoTED

```python
from poted.poted import PoTED
from main import Reporter

engine = PoTED(reporter=Reporter)

data = {"numbers": [1, 2, 3], "nested": {"flag": True}}
tensor = engine(data)
result = engine.decode(tensor)
```

```python
import numpy as np
import torch
from poted.poted import PoTED

engine = PoTED()

tensor_np = engine(np.array([1, 2, 3]))
roundtrip_np = engine.decode(tensor_np)

tensor_torch = engine(torch.tensor([1, 2, 3]))
roundtrip_torch = engine.decode(tensor_torch)
```

## Configuration options

`PoTED` accepts several options:

- `Lw`, `Le`, `Lu`: limits for dictionary and token sizes.
- `mode`: `'instance'` or `'portable'`.
- `device`: target device for tensors, e.g. `'cpu'` or `'cuda'`.
- `persistent`: keep dictionary state across calls.
- `serializer`, `tokenizer`, `decoder`, `tensor_builder`: custom components.
- `reporter`: metrics collector.

These can also be overridden per call:

```python
tensor = engine(data, mode="portable")
restored = engine.decode(tensor, device="cpu")
```

## Reporter metrics

Metrics are recorded automatically:

- `total_tokens`
- `compression_ratio`
- `tensor_shape`
- `encode_calls`
- `decode_calls`
- `sync_resets`
- `roundtrip_failures`

Query metrics via the reporter:

```python
from main import Reporter
print(Reporter.report(["total_tokens", "compression_ratio", "tensor_shape"]))
```

