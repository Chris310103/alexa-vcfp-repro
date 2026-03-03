from typing import List

def accuracy_score(sample:List[str],pre:List[str]) -> int:
    if len(sample) != len(pre):
        raise ValueError(f'length mismatch')
    if not pre:
        return 0
    correct=sum(1 for a, b in zip(sample, pre) if a==b)

    return correct / len(sample)


