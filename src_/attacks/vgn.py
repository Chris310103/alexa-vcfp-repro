import math
from typing import List
import numpy as np
from sklearn.naive_bayes import GaussianNB

from src_.attacks.base import AttackModel
from src_.data.schema import Trace


class vgn_model(AttackModel):
    def __init__(
        self,
        rounding: int | None = 5000,   
        start: int | None = None,
        end: int | None = None,
        clip_to_edges: bool = True,
    ):
        if rounding is None or not isinstance(rounding, int) or rounding <= 0:
            raise ValueError("rounding must be a positive integer")

        self.rounding = rounding
        self.start = start
        self.end = end
        self.clip_to_edges = clip_to_edges

        self.model = GaussianNB()
        self.bin_edges_ = None
        self.n_bins_ = None
        self.is_fitted_ = False

    def calculate_bursts(self, trace: Trace) -> List[int]:
        if not trace.packets:
            return []

        bursts = []
        direction = int(trace.packets[0].d)
        tmp_burst = abs(int(trace.packets[0].l)) * direction

        for i, p in enumerate(trace.packets):
            if i == 0:
                continue

            plen = abs(int(p.l))
            tmp_direc = int(p.d)
            signed_len = plen * tmp_direc

            if direction != tmp_direc:
                bursts.append(tmp_burst)
                direction = tmp_direc
                tmp_burst = signed_len
            else:
                tmp_burst += signed_len

        bursts.append(tmp_burst)
        return bursts

    def _round_down_to_multiple(self, x: int) -> int:
        return (x // self.rounding) * self.rounding

    def _round_up_to_multiple(self, x: int) -> int:
        return ((x + self.rounding - 1) // self.rounding) * self.rounding

    def _learn_range_from_train(self, traces: List[Trace]):
        all_bursts = []
        for tr in traces:
            all_bursts.extend(self.calculate_bursts(tr))

        if not all_bursts:
            self.start = -self.rounding
            self.end = self.rounding
            return

        min_b = min(all_bursts)
        max_b = max(all_bursts)

        start = self._round_down_to_multiple(min_b) - self.rounding
        end = self._round_up_to_multiple(max_b) + self.rounding

        if self.start is None:
            self.start = start
        if self.end is None:
            self.end = end

    def _build_bin_edges(self):
        edges = list(range(self.start, self.end + self.rounding, self.rounding))
        if len(edges) < 2:
            raise ValueError("invalid bin edges")
        self.bin_edges_ = np.array(edges, dtype=float)
        self.n_bins_ = len(self.bin_edges_) - 1

    def _burst_to_bin_index(self, value: int) -> int:
        idx = np.searchsorted(self.bin_edges_, value, side="right") - 1

        if self.clip_to_edges:
            if idx < 0:
                return 0
            if idx >= self.n_bins_:
                return self.n_bins_ - 1
            return int(idx)

        if idx < 0 or idx >= self.n_bins_:
            return None
        return int(idx)

    def compute_feature(self, trace: Trace) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model is not fitted yet")

        if not trace.packets:
            return np.zeros(3 + self.n_bins_, dtype=float)

        up_stream_total = 0
        down_stream_total = 0
        trace_time_list = []

        for p in trace.packets:
            plen = abs(int(p.l))
            direc = int(p.d)

            trace_time_list.append(float(p.t))

            if direc == 1:
                up_stream_total += plen
            elif direc == -1:
                down_stream_total += plen
            else:
                raise ValueError(f"unexpected direction value: {direc}")

        bursts = self.calculate_bursts(trace)

        section_list = np.zeros(self.n_bins_, dtype=float)
        for feat in bursts:
            idx = self._burst_to_bin_index(feat)
            if idx is not None:
                section_list[idx] += 1.0

        trace_time_list.sort()
        total_trace_time = trace_time_list[-1] - trace_time_list[0] if len(trace_time_list) >= 2 else 0.0

        feat_vec = [total_trace_time, float(up_stream_total), float(down_stream_total)]
        feat_vec.extend(section_list.tolist())
        return np.array(feat_vec, dtype=float)

    def vectorize_traces(self, traces: List[Trace]) -> np.ndarray:
        X = [self.compute_feature(tr) for tr in traces]
        return np.array(X, dtype=float)

    def fit(self, traces: List[Trace]):
        if not traces:
            raise ValueError("empty traces")

        self._learn_range_from_train(traces)
        self._build_bin_edges()
        self.is_fitted_ = True

        X = self.vectorize_traces(traces)
        y = [tr.label for tr in traces]

        self.model.fit(X, y)

    def predict_one(self, trace: Trace):
        x = self.compute_feature(trace).reshape(1, -1)
        return self.model.predict(x)[0]

    


if __name__ == "__main__":
    from pathlib import Path
    from src_.data.loader import loader_all_trace
    from src_.data.split import stratified_split_by_label
    from src_.eval.metrics import accuracy_score
    pp = Path(__file__).resolve().parent.parent.parent
    trace_dir = pp / "external" / "trace_csv"
    model=vgn_model(rounding=5000)
    traces = loader_all_trace(trace_dir)
    
    train_data, test_data = stratified_split_by_label(traces, seed=0)
    model.fit(train_data)

    train_true = [tr.label for tr in train_data]
    test_true = [tr.label for tr in test_data]

    train_pred = model.predict(train_data)
    test_pred = model.predict(test_data)

    print("train acc:", accuracy_score(train_true, train_pred))
    print("test acc:", accuracy_score(test_true, test_pred))

    

    


        
        
