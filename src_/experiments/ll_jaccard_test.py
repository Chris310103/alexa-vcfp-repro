from pathlib import Path
from collections import Counter

from src_.data.loader import loader_all_trace
from src_.data.split import stratified_split_by_label
from src_.data.schema import Trace
from src_.attacks.ll_jaccard import lljaccardModel
from src_.eval.metrics import accuracy_score

pp=Path(__file__).parent.parent
fp=pp.parent/'external'/'trace_csv'
traces=loader_all_trace(fp)

model=lljaccardModel(rounding=70)

data=stratified_split_by_label(traces)
train_data=data[0]
test_data=data[1]

model.fit(train_data)
train_label=[tr.label for tr in train_data]
test_label=[tr.label for tr in test_data]


pre_test=model.predict(test_data)
pre_train=model.predict(train_data)
train_acc=accuracy_score(train_label, pre_train)
test_acc=accuracy_score(test_label, pre_test)

print("train size:", len(train_data))
print("test size:", len(test_data))
print("train acc:", train_acc)
print("test acc:", test_acc)
print("top predicted labels:", Counter(pre_test).most_common(10))



