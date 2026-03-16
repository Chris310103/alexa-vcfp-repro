import numpy as np
from sklearn.naive_bayes import GaussianNB
from typing import Dict, Set, List
import math
from pathlib import Path
from collections import Counter

from src_.attacks.base import AttackModel
from src_.data.schema import Trace


from src_.features.ll_features import create_ll_features
from src_.data.loader import loader_all_trace
from src_.data.split import stratified_split_by_label
from src_.eval.metrics import accuracy_score


class ll_nb_Gaussian_Model(AttackModel):
    def __init__(
        self,
        rounding: int | None = None,
        start: int = -1500,
        end: int = 1500,
    ):
        self.rounding = rounding
        self.start = start
        self.end = end

        self.model=GaussianNB()
        self.bin_edges = self._build_bin_edges()

    def _build_bin_edges(self):
        if self.rounding is None or self.rounding <=0:
            raise ValueError("rounding must be a positive integer")
        edges = list(range(self.start, self.end+self.rounding, self.rounding))
        return np.array(edges)
    
    def packet_to_signed_length(self,p)-> int:
        length=abs(int(p.l))
        direction=int(p.d)
        return direction * length
    
    def _vectorize_trace(self,trace:Trace)->np.ndarray:
        signed_lengths=[self.packet_to_signed_length(p) for p in trace.packets]
        hist, edges=np.histogram(signed_lengths, bins=self.bin_edges)
        return hist.astype(float)
    
    def vectorize_traces(self,traces:Trace)->np.ndarray:
        X=[self._vectorize_trace(tr) for tr in traces]
        return np.array(X,dtype=float)

    def fit(self, traces: List[Trace]):
        if not traces:
            raise ValueError("empty training data")
        X=self.vectorize_traces(traces)
        y=[tr.label for tr in traces]
        self.model.fit(X,y)

    def predict_one(self, trace: Trace) -> str:
        x = self._vectorize_trace(trace).reshape(1, -1)
        pred = self.model.predict(x)[0]
        return pred
    
if __name__ == "__main__":
    from pathlib import Path
    from src_.data.loader import loader_all_trace
    from src_.data.split import stratified_split_by_label
    from src_.attacks.ll_nb_Gaussian import ll_nb_Gaussian_Model
    from src_.eval.metrics import accuracy_score

    pp = Path(__file__).resolve().parent.parent.parent
    trace_dir = pp / "external" / "trace_csv"

    traces = loader_all_trace(trace_dir)
    train_data, test_data = stratified_split_by_label(traces, seed=0)

    model = ll_nb_Gaussian_Model(rounding=100)
    model.fit(train_data)

    train_true = [tr.label for tr in train_data]
    test_true = [tr.label for tr in test_data]

    train_pred = model.predict(train_data)
    test_pred = model.predict(test_data)

    print("train acc:", accuracy_score(train_true, train_pred))
    print("test acc:", accuracy_score(test_true, test_pred))



