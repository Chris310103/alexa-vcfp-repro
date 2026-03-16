from src_.data.schema import Trace
from typing import List


def round_length(length:int,base:int|None=None) -> int:
    if base is None or base <= 1:
        return length
    return int(round((length+base//2) // base * base))


def create_burst(traces:Trace,rounding:int|None=None):
    if not traces:
        raise ValueError("empty traces")
    tokens=set()
    pakt=traces.packets
    count=0
    d=pakt[0].d
    for p in pakt:
        plen=abs(p.l)
        if d==p:
            count+=plen
        else:
            tokens.add(d*round_length(count,rounding))
            d=p.d
            count=plen
    tokens.add(d*round_length(count,rounding))
    return tokens