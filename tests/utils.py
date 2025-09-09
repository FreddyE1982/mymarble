import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import main
from poted.tokenizer import StreamingTokenizer
from poted.pipeline import JsonSerializer, TensorBuilder, PoTEDPipeline


def roundtrip(obj):
    serializer = JsonSerializer()
    tokenizer = StreamingTokenizer(main.Reporter)
    builder = TensorBuilder()
    pipeline = PoTEDPipeline(serializer, tokenizer, builder, main.Reporter)
    try:
        tensor = pipeline.encode(obj)
        result = pipeline.decode(tensor)
        if result != obj:
            count = main.Reporter.report('roundtrip_failures') or 0
            main.Reporter.report(
                'roundtrip_failures', 'Number of failed roundtrip comparisons', count + 1
            )
        return result
    except Exception:
        count = main.Reporter.report('roundtrip_failures') or 0
        main.Reporter.report(
            'roundtrip_failures', 'Number of failed roundtrip comparisons', count + 1
        )
        raise
