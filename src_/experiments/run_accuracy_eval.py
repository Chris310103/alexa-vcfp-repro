from pathlib import Path
from collections import Counter, defaultdict
import csv

from src_.data.loader import loader_all_trace
from src_.data.split import iter_stratified_kfold_by_label, summarize_dataset
from src_.features.ll_features import create_ll_features
from src_.eval.metrics import accuracy_score

from src_.attacks.ll_jaccardNN import lljaccard_NNModel
from src_.attacks.ll_jaccard_classset import lljaccard_classsetModel
from src_.attacks.ll_nb_Bernoulli import ll_nb_Model
from src_.attacks.ll_nb_Gaussian import ll_nb_Gaussian_Model
from src_.attacks.vgn import vgn_model
from src_.attacks.svm import svm_model


def mean_std(values):
    if not values:
        return 0.0, 0.0
    mean_val = sum(values) / len(values)
    if len(values) == 1:
        return mean_val, 0.0
    var = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return mean_val, var ** 0.5


def mean_oov_ratio(test_traces, vocab, rounding=None):
    if not test_traces:
        return 0.0
    ratios = []
    for tr in test_traces:
        feats = create_ll_features(tr, rounding=rounding)
        if not feats:
            ratios.append(0.0)
            continue
        oov = sum(1 for tk in feats if tk not in vocab)
        ratios.append(oov / len(feats))
    return sum(ratios) / len(ratios)


def build_model(model_name, rounding=None, alpha=None, use_unk=False):
    if model_name == "jaccard_NN":
        return lljaccard_NNModel(rounding=rounding)
    if model_name == "jaccard_class_set":
        return lljaccard_classsetModel(rounding=rounding)
    if model_name == "nb_bernoulli":
        return ll_nb_Model(rounding=rounding, alpha=alpha, use_unk=use_unk)
    if model_name == "nb_gaussian":
        return ll_nb_Gaussian_Model(rounding=rounding)
    if model_name == "vgn":
        return vgn_model(rounding=rounding)
    if model_name == "svm_linear":
        return svm_model(rounding=rounding, kernel_mode=0)
    raise ValueError(f"unknown model: {model_name}")


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one_fold(model_name, rounding, train_data, test_data, seed, fold, alpha=None, use_unk=False):
    model = build_model(
        model_name=model_name,
        rounding=rounding,
        alpha=alpha,
        use_unk=use_unk,
    )
    model.fit(train_data)

    train_true = [tr.label for tr in train_data]
    test_true = [tr.label for tr in test_data]

    train_pred = model.predict(train_data)
    test_pred = model.predict(test_data)

    train_acc = accuracy_score(train_true, train_pred)
    test_acc = accuracy_score(test_true, test_pred)

    metric_row = {
        "model": model_name,
        "rounding": rounding,
        "alpha": alpha if alpha is not None else "",
        "use_unk": use_unk,
        "defense": "",
        "d": "",
        "rho": "",
        "tau": "",
        "seed": seed,
        "fold": fold,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "oov_ratio": "",
        "top_predicted_labels": str(Counter(test_pred).most_common(10)),
    }

    if model_name == "nb_bernoulli":
        metric_row["oov_ratio"] = mean_oov_ratio(test_data, model.vocab, rounding)

    prediction_rows = []
    for tr, pred in zip(test_data, test_pred):
        prediction_rows.append({
            "trace_id": tr.trace_id,
            "true_label": tr.label,
            "pred_label": pred,
            "model": model_name,
            "rounding": rounding,
            "alpha": alpha if alpha is not None else "",
            "use_unk": use_unk,
            "defense": "",
            "d": "",
            "rho": "",
            "tau": "",
            "seed": seed,
            "fold": fold,
        })

    return metric_row, prediction_rows


def aggregate_seed(metrics_rows):
    grouped_seed = defaultdict(list)
    for row in metrics_rows:
        key = (
            row["model"],
            row["rounding"],
            row["alpha"],
            row["use_unk"],
            row["defense"],
            row["d"],
            row["rho"],
            row["tau"],
            row["seed"],
        )
        grouped_seed[key].append(row)

    seed_rows = []
    for key, rows in grouped_seed.items():
        model, rounding, alpha, use_unk, defense, d, rho, tau, seed = key

        train_list = [r["train_acc"] for r in rows]
        test_list = [r["test_acc"] for r in rows]
        oov_vals = [r["oov_ratio"] for r in rows if r["oov_ratio"] != ""]

        train_mean, train_std = mean_std(train_list)
        test_mean, test_std = mean_std(test_list)
        oov_mean = sum(oov_vals) / len(oov_vals) if oov_vals else ""

        seed_rows.append({
            "model": model,
            "rounding": rounding,
            "alpha": alpha,
            "use_unk": use_unk,
            "defense": defense,
            "d": d,
            "rho": rho,
            "tau": tau,
            "seed": seed,
            "train_acc_mean": train_mean,
            "train_acc_std": train_std,
            "test_acc_mean": test_mean,
            "test_acc_std": test_std,
            "oov_mean": oov_mean,
        })
    return seed_rows


def aggregate_overall(seed_rows):
    grouped_overall = defaultdict(list)
    for row in seed_rows:
        key = (
            row["model"],
            row["rounding"],
            row["alpha"],
            row["use_unk"],
            row["defense"],
            row["d"],
            row["rho"],
            row["tau"],
        )
        grouped_overall[key].append(row)

    overall_rows = []
    for key, rows in grouped_overall.items():
        model, rounding, alpha, use_unk, defense, d, rho, tau = key

        train_means = [r["train_acc_mean"] for r in rows]
        test_means = [r["test_acc_mean"] for r in rows]
        oov_vals = [r["oov_mean"] for r in rows if r["oov_mean"] != ""]

        train_mean, train_std = mean_std(train_means)
        test_mean, test_std = mean_std(test_means)
        oov_mean = sum(oov_vals) / len(oov_vals) if oov_vals else ""

        overall_rows.append({
            "model": model,
            "rounding": rounding,
            "alpha": alpha,
            "use_unk": use_unk,
            "defense": defense,
            "d": d,
            "rho": rho,
            "tau": tau,
            "train_acc_mean_over_seeds": train_mean,
            "train_acc_std_over_seeds": train_std,
            "test_acc_mean_over_seeds": test_mean,
            "test_acc_std_over_seeds": test_std,
            "oov_mean_over_seeds": oov_mean,
        })

    overall_rows.sort(key=lambda x: (-x["test_acc_mean_over_seeds"], x["model"]))
    return overall_rows


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    trace_dir = project_root / "external" / "trace_csv"
    output_dir = project_root / "output" / "eval_accuracy"
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = loader_all_trace(trace_dir)
    print("dataset summary:", summarize_dataset(traces))

    seeds = [0, 1, 2, 3, 4, 5]
    ll_roundings = [10, 20, 30, 50, 100]
    nb_alphas = [0.1, 0.5, 1.0]
    vgn_roundings = [1000, 2000, 5000, 8000, 10000]
    svm_roundings = [1000, 2000, 5000, 8000, 10000]

    metrics_rows = []
    predictions_rows = []

    exp_plan = [
        ("jaccard_NN", ll_roundings, [None]),
        ("jaccard_class_set", ll_roundings, [None]),
        ("nb_bernoulli", ll_roundings, nb_alphas),
        ("nb_gaussian", ll_roundings, [None]),
        ("vgn", vgn_roundings, [None]),
        ("svm_linear", svm_roundings, [None]),
    ]

    for model_name, rounding_list, alpha_list in exp_plan:
        for seed in seeds:
            for rounding in rounding_list:
                for alpha in alpha_list:
                    for fold_idx, (train_data, test_data) in enumerate(
                        iter_stratified_kfold_by_label(traces, k=5, seed=seed)
                    ):
                        metric_row, pred_rows = run_one_fold(
                            model_name=model_name,
                            rounding=rounding,
                            train_data=train_data,
                            test_data=test_data,
                            seed=seed,
                            fold=fold_idx,
                            alpha=alpha,
                        )
                        metrics_rows.append(metric_row)
                        predictions_rows.extend(pred_rows)

                        msg = (
                            f"[ACC] model={model_name} seed={seed} rounding={rounding} "
                            f"alpha={alpha} fold={fold_idx} "
                            f"train={metric_row['train_acc']:.4f} test={metric_row['test_acc']:.4f}"
                        )
                        if model_name == "nb_bernoulli":
                            msg += f" oov={metric_row['oov_ratio']}"
                        print(msg)

    seed_rows = aggregate_seed(metrics_rows)
    overall_rows = aggregate_overall(seed_rows)

    write_csv(
        output_dir / "accuracy_metrics.csv",
        fieldnames=[
            "model", "rounding", "alpha", "use_unk",
            "defense", "d", "rho", "tau",
            "seed", "fold",
            "train_acc", "test_acc", "oov_ratio",
            "top_predicted_labels",
        ],
        rows=metrics_rows,
    )

    write_csv(
        output_dir / "accuracy_predictions.csv",
        fieldnames=[
            "trace_id", "true_label", "pred_label",
            "model", "rounding", "alpha", "use_unk",
            "defense", "d", "rho", "tau",
            "seed", "fold",
        ],
        rows=predictions_rows,
    )

    write_csv(
        output_dir / "accuracy_seed_summary.csv",
        fieldnames=[
            "model", "rounding", "alpha", "use_unk",
            "defense", "d", "rho", "tau", "seed",
            "train_acc_mean", "train_acc_std",
            "test_acc_mean", "test_acc_std",
            "oov_mean",
        ],
        rows=seed_rows,
    )

    write_csv(
        output_dir / "accuracy_overall.csv",
        fieldnames=[
            "model", "rounding", "alpha", "use_unk",
            "defense", "d", "rho", "tau",
            "train_acc_mean_over_seeds", "train_acc_std_over_seeds",
            "test_acc_mean_over_seeds", "test_acc_std_over_seeds",
            "oov_mean_over_seeds",
        ],
        rows=overall_rows,
    )

    print("saved:")
    print(output_dir / "accuracy_metrics.csv")
    print(output_dir / "accuracy_predictions.csv")
    print(output_dir / "accuracy_seed_summary.csv")
    print(output_dir / "accuracy_overall.csv")


if __name__ == "__main__":
    main()