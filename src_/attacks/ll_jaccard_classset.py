from collections import defaultdict
from typing import List, Set, Tuple
from pathlib import Path


from src_.data.schema import Trace
from src_.features.ll_features import create_ll_features
from src_.attacks.base import AttackModel
from src_.data.loader import loader_all_trace
from src_.data.split import stratified_split_by_label

class lljaccard_classsetModel(AttackModel):
    def __init__(self,rounding:int|None=None,):
        self.rounding=rounding
        self.class_set: List[Tuple[str,Set[int]]]=[]

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
        class_set=[]
        bucket=defaultdict(list)
        for tr in traces:
            bucket[tr.label].append(tr)
        
        for label, item in bucket.items():
            tokens=set()
            for tr in item:
                token=create_ll_features(tr,rounding=self.rounding)
                tokens|=token
            class_set.append((label,tokens))
        self.class_set=class_set

    def predict_one(self,trace:Trace) -> str:
        if not self.class_set:
            raise RuntimeError("model is not fitted yet")
        y=create_ll_features(trace,rounding=self.rounding)
        best_label=None
        best_score=-1.0
        for feature in self.class_set:
            class_set=feature[1]
            score=self.jaccard(y,class_set)
            if score > best_score:
                best_score=score
                best_label=feature[0]
        return best_label


