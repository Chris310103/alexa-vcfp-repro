from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from src_.eval.semantic import (
    add_config_columns,
    append_semantic_metrics,
    build_semantic_tables,
    summarize_overall,
    summarize_seed_level,
)


DEFAULT_SEED_GROUP_COLS = [
    "experiment",
    "model",
    "rounding",
    "alpha",
    "use_unk",
    "defense",
    "d",
    "rho",
    "tau",
    "seed",
    "fold",
]

DEFAULT_OVERALL_GROUP_COLS = [
    "experiment",
    "model",
    "rounding",
    "alpha",
    "use_unk",
    "defense",
    "d",
    "rho",
    "tau",
]

SERIES_ORDER = [
    "jaccard_NN",
    "jaccard_class_set",
    "nb_bernoulli | alpha=0.1",
    "nb_bernoulli | alpha=0.5",
    "nb_bernoulli | alpha=1.0",
    "nb_gaussian",
    "svm_linear",
    "vgn",
]

MODEL_ORDER = [
    "jaccard_NN",
    "jaccard_class_set",
    "nb_bernoulli",
    "nb_gaussian",
    "svm_linear",
    "vgn",
]

EXPERIMENT_ORDER = ["baseline", "buflo"]


def _read_prediction_file(path: str, experiment_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["experiment"] = experiment_name
    return df


def load_predictions(
    accuracy_pred: Optional[str],
    buflo_pred: Optional[str],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if accuracy_pred:
        frames.append(_read_prediction_file(accuracy_pred, "baseline"))
    if buflo_pred:
        frames.append(_read_prediction_file(buflo_pred, "buflo"))
    if not frames:
        raise ValueError("At least one of --accuracy-pred or --buflo-pred must be provided")
    return pd.concat(frames, ignore_index=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_label_semantic_tables(output_dir: Path, tables) -> None:
    labels_df = pd.DataFrame(
        {
            "label": tables.labels,
            "command_text": tables.command_texts,
        }
    )
    labels_df.to_csv(output_dir / "semantic_label_texts.csv", index=False)

    sim_df = pd.DataFrame(tables.similarity, index=tables.labels, columns=tables.labels)
    rank_df = pd.DataFrame(tables.rank, index=tables.labels, columns=tables.labels)
    sim_df.to_csv(output_dir / "semantic_similarity_matrix.csv")
    rank_df.to_csv(output_dir / "semantic_rank_matrix.csv")


def _format_rounding_label(x) -> str:
    if pd.isna(x):
        return ""
    try:
        x = float(x)
        if x.is_integer():
            return str(int(x))
        return str(x)
    except Exception:
        return str(x)


def _series_sort_key(series_label: str) -> tuple[int, str]:
    if series_label in SERIES_ORDER:
        return (SERIES_ORDER.index(series_label), series_label)
    return (10_000, series_label)


def _model_sort_key(model_name: str) -> tuple[int, str]:
    if model_name in MODEL_ORDER:
        return (MODEL_ORDER.index(model_name), model_name)
    return (10_000, model_name)


def _experiment_sort_key(experiment_name: str) -> tuple[int, str]:
    if experiment_name in EXPERIMENT_ORDER:
        return (EXPERIMENT_ORDER.index(experiment_name), experiment_name)
    return (10_000, experiment_name)


def _ordered_experiments(df: pd.DataFrame) -> list[str]:
    exps = df["experiment"].dropna().unique().tolist()
    return sorted(exps, key=_experiment_sort_key)


def _ordered_models(df: pd.DataFrame) -> list[str]:
    models = df["model"].dropna().unique().tolist()
    return sorted(models, key=_model_sort_key)


def _ordered_series(df: pd.DataFrame) -> list[str]:
    series = df["series_label"].dropna().unique().tolist()
    return sorted(series, key=_series_sort_key)


def _best_per_series(
    overall_df: pd.DataFrame,
    metric_col: str = "accuracy_mean",
    ascending: bool = False,
) -> pd.DataFrame:
    if overall_df.empty:
        return overall_df.copy()
    plot_df = add_config_columns(overall_df.copy())
    if metric_col not in plot_df.columns:
        return plot_df.iloc[0:0].copy()
    best_df = (
        plot_df.sort_values(metric_col, ascending=ascending)
        .groupby(["experiment", "series_label"], dropna=False, as_index=False)
        .head(1)
        .copy()
    )
    return best_df


def _save_best_bar_by_experiment(
    best_df: pd.DataFrame,
    metric_col: str,
    xlabel: str,
    title: str,
    output_path: Path,
) -> None:
    if best_df.empty or metric_col not in best_df.columns:
        return

    experiments = _ordered_experiments(best_df)
    if not experiments:
        return

    fig, axes = plt.subplots(
        1,
        len(experiments),
        figsize=(8 * len(experiments), max(6, 0.55 * max(best_df.groupby("experiment").size().tolist()))),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, experiment in zip(axes, experiments):
        exp_df = best_df[best_df["experiment"] == experiment].copy()
        if exp_df.empty:
            ax.axis("off")
            continue
        exp_df = exp_df.sort_values(metric_col, ascending=True)
        ax.barh(exp_df["series_label"], exp_df[metric_col])
        ax.set_title(experiment)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Series")

    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_best_config_bars(overall_df: pd.DataFrame, plots_dir: Path) -> None:
    if overall_df.empty:
        return

    best_acc_df = _best_per_series(overall_df, metric_col="accuracy_mean", ascending=False)
    _save_best_bar_by_experiment(
        best_df=best_acc_df,
        metric_col="accuracy_mean",
        xlabel="Accuracy",
        title="Best config per experiment/series: accuracy",
        output_path=plots_dir / "best_config_accuracy.png",
    )

    nsd_col = "normalized_semantic_distance_error_mean"
    if nsd_col in overall_df.columns:
        best_nsd_df = _best_per_series(overall_df, metric_col=nsd_col, ascending=False)
        _save_best_bar_by_experiment(
            best_df=best_nsd_df,
            metric_col=nsd_col,
            xlabel="Normalized semantic distance (errors only)",
            title="Best config per experiment/series: error-only normalized semantic distance",
            output_path=plots_dir / "best_config_error_nsd.png",
        )


def plot_accuracy_vs_semantic_scatter(overall_df: pd.DataFrame, plots_dir: Path) -> None:
    if overall_df.empty:
        return
    if "accuracy_mean" not in overall_df.columns or "normalized_semantic_distance_error_mean" not in overall_df.columns:
        return

    plot_df = _best_per_series(overall_df, metric_col="accuracy_mean", ascending=False)
    plot_df = plot_df.dropna(subset=["accuracy_mean", "normalized_semantic_distance_error_mean"]).copy()
    if plot_df.empty:
        return

    experiments = _ordered_experiments(plot_df)
    if not experiments:
        return

    fig, axes = plt.subplots(
        1,
        len(experiments),
        figsize=(8 * len(experiments), 6),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, experiment in zip(axes, experiments):
        exp_df = plot_df[plot_df["experiment"] == experiment].copy()
        for series_label in _ordered_series(exp_df):
            series_df = exp_df[exp_df["series_label"] == series_label]
            if series_df.empty:
                continue
            ax.scatter(
                series_df["accuracy_mean"],
                series_df["normalized_semantic_distance_error_mean"],
                s=70,
                label=series_label,
            )
        ax.set_title(experiment)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Normalized semantic distance (errors only)")
        ax.legend(fontsize=8, loc="best")

    plt.suptitle("Accuracy vs semantic leakage distance (best accuracy config per series)", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_vs_error_nsd_scatter.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_by_rounding_grid(
    overall_df: pd.DataFrame,
    plots_dir: Path,
    metric_col: str,
    ylabel: str,
    filename_prefix: str,
) -> None:
    if overall_df.empty or metric_col not in overall_df.columns or "rounding" not in overall_df.columns:
        return

    plot_df = add_config_columns(overall_df.copy())
    plot_df["rounding_num"] = pd.to_numeric(plot_df["rounding"], errors="coerce")
    plot_df = plot_df[plot_df["rounding_num"].notna()].copy()
    if plot_df.empty:
        return

    for experiment in _ordered_experiments(plot_df):
        exp_df = plot_df[plot_df["experiment"] == experiment].copy()
        models = _ordered_models(exp_df)
        if not models:
            continue

        ncols = 2
        nrows = math.ceil(len(models) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(14, 4.8 * nrows),
            squeeze=False,
        )
        axes = axes.ravel()

        for ax, model in zip(axes, models):
            model_df = exp_df[exp_df["model"] == model].copy()
            if model_df.empty:
                ax.axis("off")
                continue

            model_roundings = sorted(model_df["rounding_num"].dropna().unique().tolist())
            x_map = {r: i for i, r in enumerate(model_roundings)}
            xticks = list(range(len(model_roundings)))
            xtick_labels = [_format_rounding_label(r) for r in model_roundings]

            series_in_model = _ordered_series(model_df)
            for series_label in series_in_model:
                series_df = model_df[model_df["series_label"] == series_label].copy()
                if series_df.empty:
                    continue
                series_df = series_df.sort_values("rounding_num")
                series_df["rounding_x"] = series_df["rounding_num"].map(x_map)
                ax.plot(
                    series_df["rounding_x"],
                    series_df[metric_col],
                    marker="o",
                    linewidth=1.8,
                    label=series_label,
                )

            ax.set_title(model)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
            ax.set_xlabel("Rounding")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)

            if len(series_in_model) > 1:
                ax.legend(fontsize=8, loc="best")

        for ax in axes[len(models):]:
            ax.axis("off")

        plt.suptitle(f"{ylabel} by rounding ({experiment})", y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{filename_prefix}_{experiment}.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def plot_by_rounding(overall_df: pd.DataFrame, plots_dir: Path) -> None:
    _plot_metric_by_rounding_grid(
        overall_df=overall_df,
        plots_dir=plots_dir,
        metric_col="accuracy_mean",
        ylabel="Accuracy",
        filename_prefix="accuracy_by_rounding",
    )
    _plot_metric_by_rounding_grid(
        overall_df=overall_df,
        plots_dir=plots_dir,
        metric_col="normalized_semantic_distance_error_mean",
        ylabel="Error-only normalized semantic distance",
        filename_prefix="error_nsd_by_rounding",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy-pred", type=str, default=None)
    parser.add_argument("--buflo-pred", type=str, default=None)
    parser.add_argument("--label-map", type=str, default=None)
    parser.add_argument("--text-col", type=str, default=None)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "sbert", "tfidf"])
    parser.add_argument(
        "--sbert-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"

    _ensure_dir(output_dir)
    _ensure_dir(plots_dir)

    pred_df = load_predictions(args.accuracy_pred, args.buflo_pred)

    all_labels = pd.concat(
        [pred_df["true_label"], pred_df["pred_label"]],
        ignore_index=True,
    ).dropna().tolist()

    tables = build_semantic_tables(
        all_labels,
        label_map_path=args.label_map,
        text_col=args.text_col,
        backend=args.backend,
        sbert_model=args.sbert_model,
    )

    save_label_semantic_tables(output_dir, tables)

    metrics_df = append_semantic_metrics(pred_df, tables)
    metrics_path = output_dir / "semantic_distance_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    seed_summary_df = summarize_seed_level(metrics_df, DEFAULT_SEED_GROUP_COLS)
    seed_summary_df = add_config_columns(seed_summary_df)
    seed_summary_path = output_dir / "semantic_distance_seed_summary.csv"
    seed_summary_df.to_csv(seed_summary_path, index=False)

    overall_df = summarize_overall(seed_summary_df, DEFAULT_OVERALL_GROUP_COLS)
    overall_df = add_config_columns(overall_df)
    overall_path = output_dir / "semantic_distance_overall.csv"
    overall_df.to_csv(overall_path, index=False)

    meta_df = pd.DataFrame(
        [
            {
                "backend_used": tables.backend_used,
                "n_classes": tables.n_classes,
                "accuracy_pred": args.accuracy_pred,
                "buflo_pred": args.buflo_pred,
                "label_map": args.label_map,
                "text_col": args.text_col,
            }
        ]
    )
    meta_df.to_csv(output_dir / "semantic_distance_run_meta.csv", index=False)

    plot_best_config_bars(overall_df, plots_dir)
    plot_accuracy_vs_semantic_scatter(overall_df, plots_dir)
    plot_by_rounding(overall_df, plots_dir)

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved seed summary to: {seed_summary_path}")
    print(f"Saved overall summary to: {overall_path}")
    print(f"Saved plots to: {plots_dir}")
    print(f"Semantic backend used: {tables.backend_used}")
    print(f"Number of classes: {tables.n_classes}")


if __name__ == "__main__":
    main()