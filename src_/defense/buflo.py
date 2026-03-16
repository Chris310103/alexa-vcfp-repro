import math
import random
from dataclasses import dataclass
from statistics import mean
from typing import List, Dict, Any, Tuple

from src_.data.schema import Packet, Trace


@dataclass
class BuFLOStats:
    trace_id: str
    label: str
    original_packets: int
    defended_packets: int
    original_bytes: int
    defended_bytes: int
    overhead_bytes: int
    overhead_kb: float
    overhead_pct: float
    original_start_time: float
    original_end_time: float
    original_duration: float
    defended_start_time: float
    defended_end_time: float
    defended_duration: float
    time_delay: float
    d: int
    rho: int
    tau: float


def _validate_params(d: int, rho: int, tau: float) -> None:
    if not isinstance(d, int) or d <= 0:
        raise ValueError("d must be a positive integer")
    if not isinstance(rho, int) or rho <= 0:
        raise ValueError("rho must be a positive integer")
    if tau < 0:
        raise ValueError("tau must be >= 0")


def _packet_chunks(size: int, d: int) -> List[int]:
    """
    Split one real packet of 'size' bytes into chunks of size d.
    Returned list contains the REAL payload bytes carried by each emitted packet.
    Example: size=2300, d=1000 -> [1000, 1000, 300]
    """
    if size <= 0:
        return []

    chunks = []
    left = size
    while left > 0:
        take = min(left, d)
        chunks.append(take)
        left -= take
    return chunks


def buflo_trace(
    trace: Trace,
    d: int = 1000,
    rho: int = 50,
    tau: float = 20.0,
    rng: random.Random | None = None,
) -> Tuple[Trace, BuFLOStats]:
    """
    Ordered BuFLO for one trace.

    Parameters
    ----------
    trace : Trace
        Original trace
    d : int
        Fixed packet size in bytes
    rho : int
        Fixed sending rate in packets / second
    tau : float
        Minimum communication duration in seconds
    rng : random.Random | None
        Optional RNG for reproducible dummy directions

    Returns
    -------
    defended_trace : Trace
    stats : BuFLOStats
    """
    _validate_params(d, rho, tau)
    rng = rng or random.Random()

    # Empty trace handling
    if not trace.packets:
        defended_trace = Trace(
            trace_id=trace.trace_id,
            label=trace.label,
            packets=[],
        )
        stats = BuFLOStats(
            trace_id=trace.trace_id,
            label=trace.label,
            original_packets=0,
            defended_packets=0,
            original_bytes=0,
            defended_bytes=0,
            overhead_bytes=0,
            overhead_kb=0.0,
            overhead_pct=0.0,
            original_start_time=0.0,
            original_end_time=0.0,
            original_duration=0.0,
            defended_start_time=0.0,
            defended_end_time=0.0,
            defended_duration=0.0,
            time_delay=0.0,
            d=d,
            rho=rho,
            tau=tau,
        )
        return defended_trace, stats

    
    original_packets = len(trace.packets)
    original_bytes = sum(abs(int(p.l)) for p in trace.packets)
    original_start_time = float(trace.packets[0].t)
    original_end_time = float(trace.packets[-1].t)
    original_duration = max(0.0, original_end_time - original_start_time)

    
    interval = 1.0 / rho
    min_total_packets = int(math.ceil(tau * rho))

    defended_packets: List[Packet] = []
    total_overhead = 0
    out_idx = 0

    for p in trace.packets:
        size = abs(int(p.l))
        direction = int(p.d)

        payload_chunks = _packet_chunks(size, d)

        for payload in payload_chunks:
            overhead = d - payload
            total_overhead += overhead

            new_t = round(original_start_time + out_idx * interval, 6)
            defended_packets.append(Packet(t=new_t, l=d, d=direction))
            out_idx += 1

    if out_idx < min_total_packets:
        for _ in range(out_idx, min_total_packets):
            direction = rng.choice([-1, 1])
            new_t = round(original_start_time + out_idx * interval, 6)
            defended_packets.append(Packet(t=new_t, l=d, d=direction))
            total_overhead += d
            out_idx += 1

    defended_trace = Trace(
        trace_id=trace.trace_id,
        label=trace.label,
        packets=defended_packets,
    )

    defended_packets_n = len(defended_packets)
    defended_bytes = defended_packets_n * d

    if defended_packets_n > 0:
        defended_start_time = float(defended_packets[0].t)
        defended_end_time = float(defended_packets[-1].t)
        defended_duration = max(0.0, defended_end_time - defended_start_time)
    else:
        defended_start_time = original_start_time
        defended_end_time = original_start_time
        defended_duration = 0.0

    time_delay = max(0.0, defended_end_time - original_end_time)
    overhead_kb = total_overhead / 1024.0
    overhead_pct = (total_overhead / original_bytes * 100.0) if original_bytes > 0 else 0.0

    stats = BuFLOStats(
        trace_id=trace.trace_id,
        label=trace.label,
        original_packets=original_packets,
        defended_packets=defended_packets_n,
        original_bytes=original_bytes,
        defended_bytes=defended_bytes,
        overhead_bytes=total_overhead,
        overhead_kb=overhead_kb,
        overhead_pct=overhead_pct,
        original_start_time=original_start_time,
        original_end_time=original_end_time,
        original_duration=original_duration,
        defended_start_time=defended_start_time,
        defended_end_time=defended_end_time,
        defended_duration=defended_duration,
        time_delay=time_delay,
        d=d,
        rho=rho,
        tau=tau,
    )

    return defended_trace, stats


def buflo_traces(
    traces: List[Trace],
    d: int = 1000,
    rho: int = 50,
    tau: float = 20.0,
    seed: int = 0,
) -> Tuple[List[Trace], List[BuFLOStats]]:
    """
    Apply BuFLO to a list of traces.
    """
    _validate_params(d, rho, tau)
    rng = random.Random(seed)

    defended = []
    stats_list = []

    for tr in traces:
        new_trace, stats = buflo_trace(tr, d=d, rho=rho, tau=tau, rng=rng)
        defended.append(new_trace)
        stats_list.append(stats)

    return defended, stats_list


def summarize_buflo_stats(stats_list: List[BuFLOStats]) -> Dict[str, Any]:
    if not stats_list:
        return {
            "n_traces": 0,
            "avg_overhead_bytes": 0.0,
            "avg_overhead_kb": 0.0,
            "avg_overhead_pct": 0.0,
            "avg_time_delay": 0.0,
            "avg_original_bytes": 0.0,
            "avg_defended_bytes": 0.0,
            "avg_original_duration": 0.0,
            "avg_defended_duration": 0.0,
        }

    return {
        "n_traces": len(stats_list),
        "avg_overhead_bytes": mean(s.overhead_bytes for s in stats_list),
        "avg_overhead_kb": mean(s.overhead_kb for s in stats_list),
        "avg_overhead_pct": mean(s.overhead_pct for s in stats_list),
        "avg_time_delay": mean(s.time_delay for s in stats_list),
        "avg_original_bytes": mean(s.original_bytes for s in stats_list),
        "avg_defended_bytes": mean(s.defended_bytes for s in stats_list),
        "avg_original_duration": mean(s.original_duration for s in stats_list),
        "avg_defended_duration": mean(s.defended_duration for s in stats_list),
    }