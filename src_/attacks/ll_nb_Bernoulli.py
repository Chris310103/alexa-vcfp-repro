from collections import defaultdict
from typing import Dict, Set, List, Tuple
import math

from src_.data.split import stratified_split_by_label
from src_.attacks.base import AttackModel
from src_.data.schema import Trace
from src_.features.ll_features import create_ll_features

class ll_nb_Model(AttackModel):
    def __init__(self,rounding:int|None=None, alpha:float|None=None, use_unk: bool = False):
        self.rounding=rounding
        self.alpha=alpha
        self.use_unk=use_unk

        self.vocab:Set[int]=set()
        self.class_log_prior:Dict[str,float]={}

        self.log_p:Dict[str, Dict[int, float]]=defaultdict(dict)
        self.log_mp:Dict[str, Dict[int, float]]=defaultdict(dict)
        self.base_absent: Dict[str, float] = {}
        self.default_log_prob:Dict[str,float]={}

    def fit(self,traces:List[Trace]):
        class_count=defaultdict(int)
        presence_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        vocab:Set[int]=set()

        for tr in traces:
            class_count[tr.label]+=1

            tokens=create_ll_features(tr,self.rounding)
            vocab |= tokens
            for i in tokens:
                presence_counts[tr.label][i]+=1

        self.vocab=vocab
        total=len(traces)
        V=len(vocab)

        class_log_prior: Dict[str, float]={}

        for label, num in class_count.items():
            class_log_prior[label]=math.log(num/total)
        self.class_log_prior=class_log_prior

        log_p:Dict[str, Dict[int, float]]=defaultdict(dict)
        log_mp:Dict[str, Dict[int, float]]=defaultdict(dict)
        default_log_prob: Dict[str, float] = {}
        base_absent:Dict[str, float]={}

        for label, num in class_count.items():
            denom=num+self.alpha * 2
            default_p=self.alpha/denom
            default_log_prob[label]=math.log(default_p)
            base_sum=0

            for tk in self.vocab:
                c=presence_counts[label].get(tk,0)
                p=(c+self.alpha)/denom
                log_p[label][tk]=math.log(p)
                log_mp[label][tk]=math.log(1-p)
                base_sum+=math.log(1-p)
            base_absent[label]=base_sum
        self.log_p=log_p
        self.log_mp=log_mp
        self.base_absent=base_absent
        self.default_log_prob=default_log_prob

    def predict_one(self, traces:Trace) -> str:
        best_prob=-math.inf
        best_label=None
        y=create_ll_features(traces,self.rounding)
        for label, items in self.class_log_prior.items():
            scores=items + self.base_absent[label]
            for tk in y:
                if tk in self.vocab:
                    scores+=(self.log_p[label][tk]-self.log_mp[label][tk])
                else:
                    if self.use_unk:
                        scores+=self.default_log_prob[label]
            if scores > best_prob:
                best_prob=scores
                best_label=label
        return best_label
    

                

        
        

            

        
        
        

