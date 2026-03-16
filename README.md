# Alexa Voice Command Fingerprinting Reproduction

A reproduction project for the paper:

**I Can Hear Your Alexa: Voice Command Fingerprinting on Smart Home Speakers**

This project reproduces the core experimental pipeline of the paper, with a focus on:

- baseline attack accuracy
- attack accuracy under **BuFLO** defense
- post-hoc **semantic leakage / semantic distance** analysis from saved predictions

The goal of this repository is not only to re-run the released traces, but also to build a clean, modular, and reproducible pipeline for feature extraction, evaluation, defense simulation, and result analysis.

---

## Project Status

This reproduction is largely complete.

### Successfully reproduced
- baseline evaluation pipeline
- BuFLO defense pipeline
- LL-NB-style reproduction using `GaussianNB`
- VNG-style reproduction using author-aligned coarse features
- per-sample prediction logging
- semantic distance analysis from saved prediction files

### Partially reproduced
- SVM line

The SVM path currently uses `LinearSVC` as a practical smoke-test replacement. The original paper reports a stronger SVM-based setup, but exact alignment is more expensive and was not prioritized over the main baseline/BuFLO reproduction.

---

## Dataset

This project uses the `trace_csv` traces released by the paper authors.

Dataset summary:

- **1000 traces**
- **100 classes**
- **10 traces per class**

Each trace is parsed into a packet-level representation and then passed through model-specific feature extraction.

---

## Implemented Models

This repository includes the following attack lines:

### `jaccard_NN`
A stronger personal variant of LL-Jaccard.  
Each training trace keeps its own token set, and test traces are classified by nearest-neighbor Jaccard similarity.

### `jaccard_class_set`
A more paper-faithful LL-Jaccard approximation.  
Training traces from the same class are unioned into a class-level token set, and test traces are compared against class templates.

### `nb_bernoulli`
A Bernoulli Naive Bayes variant implemented as a stronger personal extension.  
It performs well in the baseline setting, but collapses under BuFLO.

### `nb_gaussian`
The main paper-aligned LL-NB reproduction line.  
After checking the released code, the author implementation was found to use `GaussianNB`, so this version should be treated as the closest LL-NB reproduction.

### `vgn`
An author-style approximation of VNG++.  
Its features are aligned to the released code and include:

- total trace time
- upstream total
- downstream total
- burst histogram

### `svm_linear`
A partial SVM reproduction using `LinearSVC`.  
This is included mainly to keep the full SVM pipeline runnable, but it is **not** currently aligned with the paper’s reported SVM performance.

---

## BuFLO Defense

This project includes an ordered BuFLO-style defense implementation.

Current main setting:

- `d = 1000`
- `rho = 50`
- `tau = 20`

The defense implementation supports:

- padding small packets to fixed size
- splitting large packets into fixed-size chunks
- constant-rate re-timing
- minimum trace duration enforcement
- overhead and delay statistics

Recorded defense statistics include:

- `overhead_bytes`
- `overhead_kb`
- `overhead_pct`
- `time_delay`

---

## Repository Structure

```text
src_/
  experiments/
    run_accuracy_eval.py
    run_buflo_eval.py
    run_semantic_distance.py

  eval/
    semantic.py

  defense/
    buflo.py

output/
  eval_accuracy/
    accuracy_metrics.csv
    accuracy_predictions.csv
    accuracy_seed_summary.csv
    accuracy_overall.csv

  eval_buflo/
    buflo_metrics.csv
    buflo_predictions.csv
    buflo_seed_summary.csv
    buflo_overall.csv

  eval_semantic/
    semantic_distance_metrics.csv
    semantic_distance_seed_summary.csv
    semantic_distance_overall.csv
    semantic_similarity_matrix.csv
    semantic_rank_matrix.csv
    plots/
```
Exact file layout may differ slightly depending on your local branch, but the experiment flow is organized around these three stages: baseline, BuFLO, and semantic analysis.

## Installation

Create an environment and install the required packages.
```bash
pip install numpy pandas scikit-learn matplotlib
```
Optional, for sentence-embedding-based semantic analysis:
```bash
pip install sentence-transformers
```

## How to Run
1. **Baseline evaluation**

Run baseline attack evaluation and save both aggregate metrics and per-sample predictions:
```bash
python -m src_.experiments.run_accuracy_eval
```
**Expected outputs**:

- output/eval_accuracy/accuracy_metrics.csv

- output/eval_accuracy/accuracy_predictions.csv

- output/eval_accuracy/accuracy_seed_summary.csv

- output/eval_accuracy/accuracy_overall.csv

2. **BuFLO evaluation**

**Run attack evaluation under BuFLO defense**:
```bash
python -m src_.experiments.run_buflo_eval
```
**Expected outputs**:

- output/eval_buflo/buflo_metrics.csv

- output/eval_buflo/buflo_predictions.csv

- output/eval_buflo/buflo_seed_summary.csv

- output/eval_buflo/buflo_overall.csv

3. **Semantic distance analysis**

Semantic analysis is performed after baseline/BuFLO evaluation.
It does not rerun the attacks. Instead, it reads saved prediction files and computes semantic similarity / semantic distance over the true and predicted command labels.

Run:
```bash
python -m src_.experiments.run_semantic_distance \
  --accuracy-pred output/eval_accuracy/accuracy_predictions.csv \
  --buflo-pred output/eval_buflo/buflo_predictions.csv \
  --output-dir output/eval_semantic
```

Optional backend control:
```bash
python -m src_.experiments.run_semantic_distance \
  --accuracy-pred output/eval_accuracy/accuracy_predictions.csv \
  --buflo-pred output/eval_buflo/buflo_predictions.csv \
  --backend tfidf \
  --output-dir output/eval_semantic
```
If sentence-transformers is installed, you can also use:
```bash
python -m src_.experiments.run_semantic_distance \
  --accuracy-pred output/eval_accuracy/accuracy_predictions.csv \
  --buflo-pred output/eval_buflo/buflo_predictions.csv \
  --backend sbert \
  --output-dir output/eval_semantic
```
**Expected outputs**:

- semantic_distance_metrics.csv

- semantic_distance_seed_summary.csv

- semantic_distance_overall.csv

- semantic similarity / rank matrices

- plots for accuracy and semantic leakage trends

## Why Predictions Are Saved

A major design choice in this project is that all experiment scripts save per-sample prediction records, not only final accuracy summaries.

Each prediction row contains fields such as:

- trace_id

- true_label

- pred_label

- model

- rounding

- alpha

- defense

- d

- rho

- tau

- seed

- fold

This makes downstream semantic leakage analysis much easier, since semantic metrics can be computed directly from predictions.csv without rerunning the attack models.

## Main Results
### Baseline

Representative baseline results:

- nb_bernoulli (rounding = 10, alpha = 0.1): **37.32%**

- jaccard_NN (rounding = 10): **36.42%**

- nb_gaussian (rounding = 10): **33.85%**

- vgn (rounding = 5000): **23.92%**

- jaccard_class_set (rounding = 10): **14.37%**

### BuFLO

Representative defended results under d=1000, rho=50, tau=20:

- vgn: **~7.9**%

- nb_gaussian: **~5.4**%

- jaccard_NN: **~1.0**%

- jaccard_class_set: **~1.0**%

- nb_bernoulli: **~1.0**%

These defended results are close to the scale reported in the paper and are the strongest part of this reproduction.

## Semantic Leakage Analysis

This repository also includes a semantic leakage stage built on top of saved predictions.

Instead of only asking whether a prediction is correct, semantic analysis asks:

- if the prediction is wrong, is it still semantically close to the true command?

- does BuFLO reduce only top-1 accuracy, or also reduce semantic leakage?

The current implementation computes:

- semantic similarity

- semantic distance

- semantic rank

- normalized semantic distance

- error-only normalized semantic distance

This stage is intended as a practical post-hoc analysis layer rather than a claim of exact paper-level semantic metric replication.

## Known Limitations

- The SVM line is only a partial reproduction and currently uses LinearSVC.

- jaccard_NN and nb_bernoulli are stronger personal variants, not the most paper-faithful main lines.

- Semantic distance is implemented as a post-hoc approximation over saved predictions.

- Exact alignment with the original released code required some interpretation, especially for LL-NB and VNG++.

## Reproduction Notes

A few implementation details were especially important during reproduction:

- the released LL-NB code is effectively GaussianNB, not Bernoulli NB

- VNG++ is better aligned through code-level feature inspection than paper text alone

- saving per-sample predictions was essential for later semantic analysis

- BuFLO reproduction was the closest and most successful part of the project

## Citation

If you use this repository, please cite the original paper.
```bibtex
@inproceedings{kennedy2019alexa,
  title={I Can Hear Your Alexa: Voice Command Fingerprinting on Smart Home Speakers},
  author={Kennedy, Nolan and others},
  booktitle={Proceedings of the IEEE Conference on Communications and Network Security},
  year={2019}
}
```

## Acknowledgment

This project is a course/research reproduction based on the original paper and released traces/code from the authors. The implementation here focuses on building a clear, modular, and inspectable reproduction pipeline rather than claiming an exact line-by-line clone of the original repository.

