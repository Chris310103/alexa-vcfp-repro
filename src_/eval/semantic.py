from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_TEXT_COLS = ["command_text", "text", "utterance", "command"]


def slug_to_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_label_key(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def build_label_text_map(
    labels: Sequence[object],
    label_map_path: Optional[str] = None,
    text_col: Optional[str] = None,
) -> Dict[str, str]:
    labels = [_normalize_label_key(x) for x in labels]
    if label_map_path is None:
        return {label: slug_to_text(label) for label in labels}
    mapping_df = pd.read_csv(label_map_path)
    if "label" not in mapping_df.columns:
        raise ValueError("label_map must contain a 'label' column")
    candidate_text_cols = [text_col] if text_col else []
    candidate_text_cols.extend([c for c in DEFAULT_TEXT_COLS if c not in candidate_text_cols])
    text_column = None
    for c in candidate_text_cols:
        if c and c in mapping_df.columns:
            text_column = c
            break
    if text_column is None:
        raise ValueError(
            "label_map must contain one text column among: "
            + ", ".join(DEFAULT_TEXT_COLS)
        )
    mapping = {}
    for _, row in mapping_df.iterrows():
        key = _normalize_label_key(row["label"])
        mapping[key] = str(row[text_column]).strip()
    for label in labels:
        mapping.setdefault(label, slug_to_text(label))
    return mapping


def encode_command_texts(
    command_texts: Sequence[str],
    backend: str = "auto",
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    backend = backend.lower()
    if backend not in {"auto", "sbert", "tfidf"}:
        raise ValueError("backend must be one of: auto, sbert, tfidf")
    if backend in {"auto", "sbert"}:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(sbert_model)
            matrix = model.encode(
                list(command_texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            matrix = np.asarray(matrix, dtype=float)
            return matrix, "sbert"
        except Exception:
            if backend == "sbert":
                raise
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(list(command_texts))
    return matrix, "tfidf"


def cosine_similarity_matrix(embeddings) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(embeddings)
    sim = np.asarray(sim, dtype=float)
    sim = np.clip(sim, -1.0, 1.0)
    return sim


@dataclass
class SemanticTables:
    labels: List[str]
    command_texts: List[str]
    similarity: np.ndarray
    rank: np.ndarray
    label_to_idx: Dict[str, int]
    backend_used: str

    @property
    def n_classes(self) -> int:
        return len(self.labels)


def build_semantic_tables(
    labels: Sequence[object],
    label_map_path: Optional[str] = None,
    text_col: Optional[str] = None,
    backend: str = "auto",
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SemanticTables:
    unique_labels = sorted({_normalize_label_key(x) for x in labels})
    label_to_text = build_label_text_map(unique_labels, label_map_path=label_map_path, text_col=text_col)
    command_texts = [label_to_text[label] for label in unique_labels]
    embeddings, backend_used = encode_command_texts(command_texts, backend=backend, sbert_model=sbert_model)
    similarity = cosine_similarity_matrix(embeddings)
    order = np.argsort(-similarity, axis=1, kind="mergesort")
    rank = np.empty_like(order, dtype=int)
    for i in range(order.shape[0]):
        rank[i, order[i]] = np.arange(order.shape[1], dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    return SemanticTables(
        labels=unique_labels,
        command_texts=command_texts,
        similarity=similarity,
        rank=rank,
        label_to_idx=label_to_idx,
        backend_used=backend_used,
    )


def append_semantic_metrics(
    df: pd.DataFrame,
    tables: SemanticTables,
    true_col: str = "true_label",
    pred_col: str = "pred_label",
) -> pd.DataFrame:
    out = df.copy()
    true_keys = out[true_col].map(_normalize_label_key)
    pred_keys = out[pred_col].map(_normalize_label_key)
    true_idx = true_keys.map(tables.label_to_idx)
    pred_idx = pred_keys.map(tables.label_to_idx)
    if true_idx.isna().any():
        missing = sorted(set(true_keys[true_idx.isna()].tolist()))
        raise ValueError(f"Unknown true labels found in predictions: {missing[:10]}")
    if pred_idx.isna().any():
        missing = sorted(set(pred_keys[pred_idx.isna()].tolist()))
        raise ValueError(f"Unknown predicted labels found in predictions: {missing[:10]}")
    true_arr = true_idx.astype(int).to_numpy()
    pred_arr = pred_idx.astype(int).to_numpy()
    sim = tables.similarity[true_arr, pred_arr]
    ranks = tables.rank[true_arr, pred_arr]
    denom = max(tables.n_classes - 1, 1)
    nsd = ranks / denom
    out["true_command_text"] = [tables.command_texts[i] for i in true_arr]
    out["pred_command_text"] = [tables.command_texts[i] for i in pred_arr]
    out["semantic_similarity"] = sim
    out["semantic_distance"] = 1.0 - sim
    out["semantic_rank"] = ranks
    out["normalized_semantic_distance"] = nsd
    out["normalized_semantic_similarity"] = 1.0 - nsd
    out["is_correct"] = (true_arr == pred_arr).astype(int)
    out["is_error"] = 1 - out["is_correct"]
    out["backend_used"] = tables.backend_used
    out["n_classes"] = tables.n_classes
    return out


def _safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def summarize_seed_level(
    metrics_df: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    group_cols = [c for c in group_cols if c in metrics_df.columns]
    rows = []
    for keys, group in metrics_df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        error_group = group[group["is_error"] == 1]
        row.update(
            n_samples=int(len(group)),
            n_errors=int(group["is_error"].sum()),
            accuracy=float(group["is_correct"].mean()),
            semantic_similarity_mean=float(group["semantic_similarity"].mean()),
            semantic_distance_mean=float(group["semantic_distance"].mean()),
            semantic_rank_mean=float(group["semantic_rank"].mean()),
            normalized_semantic_distance_mean=float(group["normalized_semantic_distance"].mean()),
            normalized_semantic_similarity_mean=float(group["normalized_semantic_similarity"].mean()),
            semantic_similarity_error_mean=_safe_mean(error_group["semantic_similarity"]),
            semantic_distance_error_mean=_safe_mean(error_group["semantic_distance"]),
            semantic_rank_error_mean=_safe_mean(error_group["semantic_rank"]),
            normalized_semantic_distance_error_mean=_safe_mean(error_group["normalized_semantic_distance"]),
            normalized_semantic_similarity_error_mean=_safe_mean(error_group["normalized_semantic_similarity"]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_overall(
    seed_summary_df: pd.DataFrame,
    overall_group_cols: Sequence[str],
) -> pd.DataFrame:
    overall_group_cols = [c for c in overall_group_cols if c in seed_summary_df.columns]
    metric_name_map = {
        "accuracy": "accuracy",
        "semantic_similarity_mean": "semantic_similarity",
        "semantic_distance_mean": "semantic_distance",
        "semantic_rank_mean": "semantic_rank",
        "normalized_semantic_distance_mean": "normalized_semantic_distance",
        "normalized_semantic_similarity_mean": "normalized_semantic_similarity",
        "semantic_similarity_error_mean": "semantic_similarity_error",
        "semantic_distance_error_mean": "semantic_distance_error",
        "semantic_rank_error_mean": "semantic_rank_error",
        "normalized_semantic_distance_error_mean": "normalized_semantic_distance_error",
        "normalized_semantic_similarity_error_mean": "normalized_semantic_similarity_error",
    }
    rows = []
    for keys, group in seed_summary_df.groupby(overall_group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(overall_group_cols, keys))
        row["n_seed_fold_groups"] = int(len(group))
        row["n_samples_total"] = int(group["n_samples"].sum()) if "n_samples" in group.columns else int(len(group))
        row["n_errors_total"] = int(group["n_errors"].sum()) if "n_errors" in group.columns else int(np.nan)
        for col, out_name in metric_name_map.items():
            row[f"{out_name}_mean"] = float(group[col].mean())
            row[f"{out_name}_std"] = float(group[col].std(ddof=0))
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["experiment", "model", "rounding", "alpha", "seed", "fold"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def add_config_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    label_parts = []
    for _, row in out.iterrows():
        parts = []
        for col in ["experiment", "model", "rounding", "alpha", "defense", "d", "rho", "tau"]:
            if col in out.columns:
                val = row[col]
                if isinstance(val, float) and math.isnan(val):
                    continue
                if pd.isna(val):
                    continue
                if col == "alpha":
                    parts.append(f"alpha={val}")
                elif col == "rounding":
                    parts.append(f"r={val}")
                elif col in {"d", "rho", "tau"}:
                    parts.append(f"{col}={val}")
                elif col == "defense" and str(val).strip() == "":
                    continue
                else:
                    parts.append(str(val))
        label_parts.append(" | ".join(parts))
    out["config_label"] = label_parts
    series_labels = []
    for _, row in out.iterrows():
        parts = []
        for col in ["model", "alpha"]:
            if col in out.columns:
                val = row[col]
                if pd.isna(val):
                    continue
                if col == "alpha":
                    parts.append(f"alpha={val}")
                else:
                    parts.append(str(val))
        series_labels.append(" | ".join(parts))
    out["series_label"] = series_labels
    return out  