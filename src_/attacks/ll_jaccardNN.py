from collections import defaultdict
from typing import List, Set, Tuple
from pathlib import Path


from src_.data.schema import Trace
from src_.features.ll_features import create_ll_features
from src_.attacks.base import AttackModel
from src_.data.loader import loader_all_trace
from src_.data.split import stratified_split_by_label

class lljaccard_NNModel(AttackModel):
    def __init__(self,rounding:int|None=None,):
        self.rounding=rounding
        self.train_fingerprinting: List[Tuple[str,Set[int]]]=[]

    def jaccard(self,a:Set[int],b:Set[int]):
        if not a and not b:
            return 1.0
        union=a|b
        if not union:
            return 0
        return len(a&b)/len(union)
    
    def fit(self,traces:List[Trace]):
        if not traces:
            raise ValueError("empty training traces")
        train_fingerprinting=[]
        bucket=defaultdict(list)
        for tr in traces:
            bucket[tr.label].append(tr)
        for label, item in bucket.items():
            for tr in item:
                feature=create_ll_features(tr,rounding=self.rounding)
                train_fingerprinting.append((label,feature))
        self.train_fingerprinting=train_fingerprinting

    def predict_one(self,trace:Trace) -> str:
        if not self.train_fingerprinting:
            raise RuntimeError("model is not fitted yet")
        y=create_ll_features(trace,rounding=self.rounding)
        best_label=None
        best_score=-1.0
        for feature in self.train_fingerprinting:
            train_fingerprinting=feature[1]
            score=self.jaccard(y,train_fingerprinting)
            if score > best_score:
                best_score=score
                best_label=feature[0]
        return best_label


