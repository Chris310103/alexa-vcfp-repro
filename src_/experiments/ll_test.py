from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Set
import csv

from src_.defense.buflo import buflo_traces, summarize_buflo_stats
from src_.data.loader import loader_all_trace
from src_.data.split import iter_stratified_kfold_by_label, summarize_dataset
from src_.data.schema import Trace
from src_.features.ll_features import create_ll_features
from src_.eval.metrics import accuracy_score

from src_.attacks.ll_jaccardNN import lljaccard_NNModel
from src_.attacks.ll_nb_Bernoulli import ll_nb_Model
from src_.attacks.ll_jaccard_classset import lljaccard_classsetModel
from src_.attacks.ll_nb_Gaussian import ll_nb_Gaussian_Model
from src_.attacks.vgn import vgn_model


def mean_std(values):
    if not values:
        return 0.0, 0.0
    mean_val = sum(values) / len(values)
    if len(values) == 1:
        return mean_val, 0.0
    var = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return mean_val, var ** 0.5


def mean_oov_ratio(test_traces: List[Trace], vocab: Set[int], rounding: int | None = None):
    if not test_traces:
        return 0.0
    ratios = []
    for tr in test_traces:
        feas = create_ll_features(tr, rounding=rounding)
        if not feas:
            ratios.append(0.0)
            continue
        oov = sum(1 for tk in feas if tk not in vocab)
        ratios.append(oov / len(feas))
    return sum(ratios) / len(ratios)


def build_model(model_name: str, rounding: int | None = None, alpha: float | None = None, use_unk=False):
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
    raise ValueError(f"unknown model {model_name}")


def eval_one_fold(
    model_name,
    train_data,
    test_data,
    rounding,
    alpha=None,
    use_unk=False,
    defense=None,
    d=None,
    rho=None,
    tau=None,
    defense_seed=0,
):
    if defense == "buflo":
        train_used, train_stats = buflo_traces(
            train_data,
            d=d,
            rho=rho,
            tau=tau,
            seed=defense_seed,
        )
        test_used, test_stats = buflo_traces(
            test_data,
            d=d,
            rho=rho,
            tau=tau,
            seed=defense_seed + 1,
        )
        train_buflo_summary = summarize_buflo_stats(train_stats)
        test_buflo_summary = summarize_buflo_stats(test_stats)
    else:
        train_used = train_data
        test_used = test_data
        train_buflo_summary = None
        test_buflo_summary = None

    model = build_model(
        model_name=model_name,
        rounding=rounding,
        alpha=alpha,
        use_unk=use_unk,
    )

    model.fit(train_used)

    train_true = [tr.label for tr in train_used]
    test_true = [tr.label for tr in test_used]

    train_pred = model.predict(train_used)
    test_pred = model.predict(test_used)

    train_acc = accuracy_score(train_true, train_pred)
    test_acc = accuracy_score(test_true, test_pred)

    row = {
        "model": model_name,
        "rounding": rounding,
        "alpha": alpha if alpha is not None else "",
        "use_unk": use_unk,
        "defense": defense if defense is not None else "",
        "d": d if d is not None else "",
        "rho": rho if rho is not None else "",
        "tau": tau if tau is not None else "",
        "train_acc": train_acc,
        "test_acc": test_acc,
        "top_predicted_labels": str(Counter(test_pred).most_common(10)),
    }

    if model_name == "nb_bernoulli":
        row["oov_ratio"] = mean_oov_ratio(test_used, model.vocab, rounding)
    else:
        row["oov_ratio"] = ""

    if defense == "buflo":
        row["train_overhead_kb"] = train_buflo_summary["avg_overhead_kb"]
        row["train_overhead_pct"] = train_buflo_summary["avg_overhead_pct"]
        row["train_delay_s"] = train_buflo_summary["avg_time_delay"]
        row["test_overhead_kb"] = test_buflo_summary["avg_overhead_kb"]
        row["test_overhead_pct"] = test_buflo_summary["avg_overhead_pct"]
        row["test_delay_s"] = test_buflo_summary["avg_time_delay"]
    else:
        row["train_overhead_kb"] = ""
        row["train_overhead_pct"] = ""
        row["train_delay_s"] = ""
        row["test_overhead_kb"] = ""
        row["test_overhead_pct"] = ""
        row["test_delay_s"] = ""

    return row


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_experiments(
    fold_rows,
    traces,
    seeds,
    model_name,
    rounding_list,
    alpha_list=None,
    defense=None,
    d=None,
    rho=None,
    tau=None,
):
    if alpha_list is None:
        alpha_list = [None]

    for seed in seeds:
        for rounding in rounding_list:
            for alpha in alpha_list:
                for fold_idx, (train_data, test_data) in enumerate(
                    iter_stratified_kfold_by_label(traces, k=5, seed=seed)
                ):
                    row = eval_one_fold(
                        model_name=model_name,
                        train_data=train_data,
                        test_data=test_data,
                        rounding=rounding,
                        alpha=alpha,
                        defense=defense,
                        d=d,
                        rho=rho,
                        tau=tau,
                        defense_seed=seed,
                    )
                    row["seed"] = seed
                    row["fold"] = fold_idx
                    fold_rows.append(row)

                    msg = (
                        f"[{defense.upper() if defense else 'BASE'}] "
                        f"model={model_name} seed={seed} rounding={rounding} fold={fold_idx} "
                    )
                    if alpha is not None:
                        msg += f"alpha={alpha} "
                    msg += f"train={row['train_acc']:.4f} test={row['test_acc']:.4f}"
                    if defense == "buflo":
                        msg += f" overhead_kb={row['test_overhead_kb']} delay={row['test_delay_s']}"
                    elif model_name == "nb_bernoulli":
                        msg += f" oov={row['oov_ratio']}"
                    print(msg)


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    trace_dir = project_root / "external" / "trace_csv"
    output_dir = project_root / "output" / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = loader_all_trace(trace_dir)
    print(f"dataset summary: {summarize_dataset(traces)}")

    seeds = [0, 1, 2, 3, 4, 5]
    ll_roundings = [10, 20, 30, 50, 100]
    nb_alphas = [0.1, 0.5, 1.0]
    vgn_roundings = [1000, 2000, 5000, 8000, 10000]

    fold_rows = []

    append_experiments(
        fold_rows=fold_rows,
        traces=traces,
        seeds=seeds,
        model_name="jaccard_NN",
        rounding_list=ll_roundings,
    )

    append_experiments(
        fold_rows=fold_rows,
        traces=traces,
        seeds=seeds,
        model_name="jaccard_class_set",
        rounding_list=ll_roundings,
    )

    append_experiments(
        fold_rows=fold_rows,
        traces=traces,
        seeds=seeds,
        model_name="nb_bernoulli",
        rounding_list=ll_roundings,
        alpha_list=nb_alphas,
    )

    append_experiments(
        fold_rows=fold_rows,
        traces=traces,
        seeds=seeds,
        model_name="nb_gaussian",
        rounding_list=ll_roundings,
    )

    append_experiments(
        fold_rows=fold_rows,
        traces=traces,
        seeds=seeds,
        model_name="vgn",
        rounding_list=vgn_roundings,
    )

    buflo_d = 1000
    buflo_rho = 50
    buflo_tau = 20

    append_experiments(
        fold_rows=fold_rows,
        traces=traces,
        seeds=seeds,
        model_name="jaccard_NN",
        rounding_list=ll_roundings,
        defense="buflo",
        d=buflo_d,
        rho=buflo_rho,
        tau=buflo_tau,
    )

    append_experiments(
        fold_rows=fold_rows,
        traces=traces,
        seeds=seeds,
        model_name="jaccard_class_set",
        rounding_list=ll_roundings,
        defense="buflo",
        d=buflo_d,
        rho=buflo_rho,
        tau=buflo_tau,
    )

    append_experiments(
        fold_rows=fold_rows,
        traces=traces,
        seeds=seeds,
        model_name="nb_bernoulli",
        rounding_list=ll_roundings,
        alpha_list=nb_alphas,
        defense="buflo",
        d=buflo_d,
        rho=buflo_rho,
        tau=buflo_tau,
    )

    append_experiments(
        fold_rows=fold_rows,
        traces=traces,
        seeds=seeds,
        model_name="nb_gaussian",
        rounding_list=ll_roundings,
        defense="buflo",
        d=buflo_d,
        rho=buflo_rho,
        tau=buflo_tau,
    )

    append_experiments(
        fold_rows=fold_rows,
        traces=traces,
        seeds=seeds,
        model_name="vgn",
        rounding_list=vgn_roundings,
        defense="buflo",
        d=buflo_d,
        rho=buflo_rho,
        tau=buflo_tau,
    )

    fold_csv = output_dir / "kfold_sweep_folds.csv"
    write_csv(
        fold_csv,
        fieldnames=[
            "model", "rounding", "alpha", "use_unk",
            "defense", "d", "rho", "tau",
            "seed", "fold",
            "train_acc", "test_acc",
            "oov_ratio",
            "train_overhead_kb", "train_overhead_pct", "train_delay_s",
            "test_overhead_kb", "test_overhead_pct", "test_delay_s",
            "top_predicted_labels",
        ],
        rows=fold_rows,
    )

    grouped_seed = defaultdict(list)
    for row in fold_rows:
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
        train_overhead_kb_vals = [r["train_overhead_kb"] for r in rows if r["train_overhead_kb"] != ""]
        train_overhead_pct_vals = [r["train_overhead_pct"] for r in rows if r["train_overhead_pct"] != ""]
        train_delay_vals = [r["train_delay_s"] for r in rows if r["train_delay_s"] != ""]
        test_overhead_kb_vals = [r["test_overhead_kb"] for r in rows if r["test_overhead_kb"] != ""]
        test_overhead_pct_vals = [r["test_overhead_pct"] for r in rows if r["test_overhead_pct"] != ""]
        test_delay_vals = [r["test_delay_s"] for r in rows if r["test_delay_s"] != ""]

        oov_mean = sum(oov_vals) / len(oov_vals) if oov_vals else ""
        train_overhead_kb_mean = sum(train_overhead_kb_vals) / len(train_overhead_kb_vals) if train_overhead_kb_vals else ""
        train_overhead_pct_mean = sum(train_overhead_pct_vals) / len(train_overhead_pct_vals) if train_overhead_pct_vals else ""
        train_delay_mean = sum(train_delay_vals) / len(train_delay_vals) if train_delay_vals else ""
        test_overhead_kb_mean = sum(test_overhead_kb_vals) / len(test_overhead_kb_vals) if test_overhead_kb_vals else ""
        test_overhead_pct_mean = sum(test_overhead_pct_vals) / len(test_overhead_pct_vals) if test_overhead_pct_vals else ""
        test_delay_mean = sum(test_delay_vals) / len(test_delay_vals) if test_delay_vals else ""

        train_mean, train_std = mean_std(train_list)
        test_mean, test_std = mean_std(test_list)

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
            "train_overhead_kb_mean": train_overhead_kb_mean,
            "train_overhead_pct_mean": train_overhead_pct_mean,
            "train_delay_mean": train_delay_mean,
            "test_overhead_kb_mean": test_overhead_kb_mean,
            "test_overhead_pct_mean": test_overhead_pct_mean,
            "test_delay_mean": test_delay_mean,
        })

    seed_csv = output_dir / "kfold_sweep_seed_summary.csv"
    write_csv(
        seed_csv,
        fieldnames=[
            "model", "rounding", "alpha", "use_unk",
            "defense", "d", "rho", "tau", "seed",
            "train_acc_mean", "train_acc_std",
            "test_acc_mean", "test_acc_std",
            "oov_mean",
            "train_overhead_kb_mean", "train_overhead_pct_mean", "train_delay_mean",
            "test_overhead_kb_mean", "test_overhead_pct_mean", "test_delay_mean",
        ],
        rows=seed_rows,
    )

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
        train_overhead_kb_vals = [r["train_overhead_kb_mean"] for r in rows if r["train_overhead_kb_mean"] != ""]
        train_overhead_pct_vals = [r["train_overhead_pct_mean"] for r in rows if r["train_overhead_pct_mean"] != ""]
        train_delay_vals = [r["train_delay_mean"] for r in rows if r["train_delay_mean"] != ""]
        test_overhead_kb_vals = [r["test_overhead_kb_mean"] for r in rows if r["test_overhead_kb_mean"] != ""]
        test_overhead_pct_vals = [r["test_overhead_pct_mean"] for r in rows if r["test_overhead_pct_mean"] != ""]
        test_delay_vals = [r["test_delay_mean"] for r in rows if r["test_delay_mean"] != ""]

        oov_mean = sum(oov_vals) / len(oov_vals) if oov_vals else ""
        train_overhead_kb_mean = sum(train_overhead_kb_vals) / len(train_overhead_kb_vals) if train_overhead_kb_vals else ""
        train_overhead_pct_mean = sum(train_overhead_pct_vals) / len(train_overhead_pct_vals) if train_overhead_pct_vals else ""
        train_delay_mean = sum(train_delay_vals) / len(train_delay_vals) if train_delay_vals else ""
        test_overhead_kb_mean = sum(test_overhead_kb_vals) / len(test_overhead_kb_vals) if test_overhead_kb_vals else ""
        test_overhead_pct_mean = sum(test_overhead_pct_vals) / len(test_overhead_pct_vals) if test_overhead_pct_vals else ""
        test_delay_mean = sum(test_delay_vals) / len(test_delay_vals) if test_delay_vals else ""

        train_mean, train_std = mean_std(train_means)
        test_mean, test_std = mean_std(test_means)

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
            "train_overhead_kb_mean_over_seeds": train_overhead_kb_mean,
            "train_overhead_pct_mean_over_seeds": train_overhead_pct_mean,
            "train_delay_mean_over_seeds": train_delay_mean,
            "test_overhead_kb_mean_over_seeds": test_overhead_kb_mean,
            "test_overhead_pct_mean_over_seeds": test_overhead_pct_mean,
            "test_delay_mean_over_seeds": test_delay_mean,
        })

    overall_rows.sort(key=lambda x: (-x["test_acc_mean_over_seeds"], x["model"]))

    overall_csv = output_dir / "kfold_sweep_overall.csv"
    write_csv(
        overall_csv,
        fieldnames=[
            "model", "rounding", "alpha", "use_unk",
            "defense", "d", "rho", "tau",
            "train_acc_mean_over_seeds", "train_acc_std_over_seeds",
            "test_acc_mean_over_seeds", "test_acc_std_over_seeds",
            "oov_mean_over_seeds",
            "train_overhead_kb_mean_over_seeds", "train_overhead_pct_mean_over_seeds", "train_delay_mean_over_seeds",
            "test_overhead_kb_mean_over_seeds", "test_overhead_pct_mean_over_seeds", "test_delay_mean_over_seeds",
        ],
        rows=overall_rows,
    )

    print("\nSaved files:")
    print(fold_csv)
    print(seed_csv)
    print(overall_csv)

    print("\nTop results:")
    for row in overall_rows[:20]:
        print(row)


if __name__ == "__main__":
    main()