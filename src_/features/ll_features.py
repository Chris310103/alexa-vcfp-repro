from src_.data.schema import Trace
from typing import Set
from pathlib import Path
from src_.data.loader import loader_all_trace



def round_length(length:int,base:int|None=None) -> int:
    if base is None or base <= 1:
        return length
    return int(round(length/base)*base)

def create_ll_features(trace:Trace, rounding:int|None=None) -> Set[int]:
    tokens:Set[int]=set()
    for p in trace.packets:
        length=round_length(p.l,rounding)
        if p.d < 0:
            tokens.add(-length)
        else:
            tokens.add(length)
    return tokens

if __name__ =="__main__":
    trace_dir = Path("external/trace_csv")
    traces = loader_all_trace(trace_dir)

    tokens = create_ll_features(traces[0], rounding=None)

    print("label:", traces[0].label)
    print("num_packets:", len(traces[0].packets))
    print("num_tokens:", len(tokens))
    print("first 20 tokens:", list(tokens)[:20])

