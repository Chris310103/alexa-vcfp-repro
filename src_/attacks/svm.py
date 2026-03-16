import math
from typing import List
import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC

from src_.data.schema import Trace
from src_.attacks.base import AttackModel

class svm_model(AttackModel):
    def __init__(self,
                 rounding:int| None=5000,
                 kernel_mode: int| None=None,
                 start:int|None=None,
                 end:int|None=None,
                 clip_to_edges:bool=True,):
        if rounding is None or not isinstance(rounding, int) or rounding <= 0:
            raise ValueError('rounding must be a positive integer')
        self.rounding=rounding
        self.kernel_mode=kernel_mode
        self.start=start
        self.end=end
        self.clip_to_edges=clip_to_edges

        self.model=self.build_svm()
        self.bin_edges=None
        self.n_bins=None
        self.is_fitted=False

    def build_svm(self):
        if self.kernel_mode==0:
            return LinearSVC(max_iter=5000,dual="auto")
        if self.kernel_mode==1:
            return svm.SVC(kernel_mode="poly", degreee=3)
        if self.kernel_mode==2:
            return svm.SVC(kernel_mode="rbf")
        if self.kernel_mode==3:
            return svm.SVC(kernel="sigmoid")
        raise ValueError("kernel_mode must be among 0,1,2,3")
    
    def calculate_bursts(self, trace:Trace) -> List[int]:
        if not trace.packets:
            return []
        burst=[]
        direction=int(trace.packets[0].d)
        tem_burst=abs(int(trace.packets[0].l)) * direction
        for i in range(1,len(trace.packets)):
            plen=abs(trace.packets[i].l)
            pd=trace.packets[i].d
            signed_length=plen*pd
            if direction != pd:
                burst.append(tem_burst)
                tem_burst=plen
                direction=pd
            else:   
                tem_burst+=plen
            
        burst.append(tem_burst)
        return burst
    
    def round_down_to_multiple(self, x:int):
        return (x//self.rounding)*self.rounding
    def round_up_to_multiple(self, x:int):
        return ((x+self.rounding-1)//self.rounding) * self.rounding
    
    def learn_range_from_train(self, traces:List[Trace]):
        all_burst=[]
        for tr in traces:
            all_burst.extend(self.calculate_bursts(tr))
        if not all_burst:
            self.start=-self.rounding
            self.end=self.rounding
        
        min_b=min(all_burst)
        max_b=max(all_burst)

        start = self.round_down_to_multiple(min_b) - self.rounding
        end = self.round_up_to_multiple(max_b) + self.rounding

        if self.start is None:
            self.start=start
        if self.end is None:
            self.end=end
    
    def build_bin_edges(self):
        edges=list(range(self.start, self.end+self.rounding, self.rounding))
        if len(edges) < 2:
            raise ValueError("invalid bin edges")
        self.bin_edges=np.array(edges, dtype=float)
        self.n_bins=len(self.bin_edges)-1

    def burst_to_bin_index(self,value:int):
        idx=np.searchsorted(self.bin_edges, value,side="right")-1

        if self.clip_to_edges:
            if idx<0:
                return 0
            if idx>=self.n_bins:
                return self.n_bins-1
            return int(idx)
        if idx<0 or idx >= self.n_bins:
            return None
        return int(idx)
    
    def compute_feature(self,trace:Trace):
        if not self.is_fitted:
            raise RuntimeError("is not fitted")
        if not trace.packets:
            return np.zeros(5+self.n_bins, dtype=float)
        
        up_stream_total=0
        down_stream_total=0
        up_pack_num=0
        down_pack_num=0

        count=len(trace.packets)
        bursts=self.calculate_bursts(trace)
        burst_num=len(bursts)

        for p in trace.packets:
            plen=abs(int(p.l))
            direc=int(p.d)

            if direc==1:
                up_stream_total+=plen
                up_pack_num+=1
            elif direc==-1:
                down_stream_total+=plen
                down_pack_num+=1
            else:
                raise ValueError(f"{direc},{plen}")
            
        total_pack=down_pack_num+up_pack_num
        in_pack_ratio=down_pack_num/total_pack if total_pack>0 else 0.0
        section_list=np.zeros(self.n_bins,dtype=float)
        for feat in bursts:
            idx=self.burst_to_bin_index(feat)
            if idx is not None:
                section_list[idx]+=1.0
        feat_vec=[
            float(up_stream_total),
            float(down_stream_total),
            float(in_pack_ratio),
            float(count),
            float(burst_num),
        ]
        feat_vec.extend(section_list.tolist())
        return np.array(feat_vec, dtype=float)
    
    def vectorize_traces(self,traces:List[Trace]):
        X=[self.compute_feature(tr) for tr in traces]
        return np.array(X, dtype=float)
    
    def fit(self, traces: List[Trace]):
        if not traces:
            raise ValueError("empty traces")

        self.learn_range_from_train(traces)
        self.build_bin_edges()
        self.is_fitted= True

        X = self.vectorize_traces(traces)
        y = [tr.label for tr in traces]

        self.model.fit(X, y)

    def predict_one(self, trace: Trace):
        x = self.compute_feature(trace).reshape(1, -1)
        return self.model.predict(x)[0]
    
if __name__=="__main__":

    from pathlib import Path
    import time

    from src_.data.loader import loader_all_trace
    from src_.data.split import stratified_split_by_label
    from src_.eval.metrics import accuracy_score
    project_root = Path(__file__).resolve().parent.parent.parent
    trace_dir = project_root / "external" / "trace_csv"

    print("1) loading traces ...")
    traces = loader_all_trace(trace_dir)
    print("loaded:", len(traces))

    print("2) splitting ...")
    train_data, test_data = stratified_split_by_label(traces, seed=0)
    print("train:", len(train_data), "test:", len(test_data))

    print("3) building model ...")
    model = svm_model(
        rounding=5000,
        kernel_mode=0,   
        start=None,
        end=None,
        clip_to_edges=True,
    )

    print("4) start fitting ...")
    t0 = time.time()
    model.fit(train_data)
    t1 = time.time()
    print(f"fit done in {t1 - t0:.2f}s")

    print("5) start predicting train ...")
    train_true = [tr.label for tr in train_data]
    train_pred = model.predict(train_data)
    print("train prediction done")

    print("6) start predicting test ...")
    test_true = [tr.label for tr in test_data]
    test_pred = model.predict(test_data)
    print("test prediction done")

    print("train acc:", accuracy_score(train_true, train_pred))
    print("test acc:", accuracy_score(test_true, test_pred))