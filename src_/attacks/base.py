from abc import ABC, abstractmethod
from typing import List
from src_.data.schema import Trace

class AttackModel(ABC):
    @abstractmethod
    def fit(self, traces:List[Trace],)-> None:
        pass

    @abstractmethod
    def predict_one(slef,traces:List[Trace]) -> str:
        pass
    
    def predict(self, traces:List[Trace]) -> List[str]:
        return [self.predict_one(t) for t in traces]