#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = {0: "Low-level", 1: "Normal", 2: "High-level"}
PROBA_COLUMNS = [f"p{c}" for c in CLASS_LABELS]
METRIC_KEYS = ["accuracy", "sensitivity", "specificity", "f1_macro"]
N_SPLITS = 5
SEED = 42
DEV_CSV = "cortisol_development_data1.csv"
EXT_CSV = "cortisol_external_data.csv"
OUTDIR = Path("reproduction")
DROP_MISSING_ROWS = True


def validate_config() -> None:
    if isinstance(N_SPLITS, bool) or not isinstance(N_SPLITS, (int, np.integer)) or N_SPLITS < 2:
        raise ValueError("N_SPLITS must be an integer >= 2.")


def save_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, encoding="utf-8-sig", float_format="%.17g", na_rep="NaN")


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load(path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path}: CSV is empty or has no header.") from exc
    raw.columns = raw.columns.str.strip()
    if raw.columns.duplicated().any():
        raise ValueError(f"{path}: duplicate column names after stripping whitespace.")
    missing = {"delta_i", "label"} - set(raw.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}.")
    frame = raw[["delta_i", "label"]].copy()
    frame.insert(0, "source_index", np.arange(len(frame)))
    for column in ("delta_i", "label"):
        values = pd.to_numeric(frame[column], errors="coerce")
        bad_text = frame[column].notna() & values.isna()
        if bad_text.any():
            bad = frame.loc[bad_text, ["source_index", column]].head(8).to_dict("records")
            raise ValueError(f"{path}: nonnumeric {column}; records: {bad}")
        infinite = values.notna() & ~np.isfinite(values)
        if infinite.any():
            rows = frame.loc[infinite, "source_index"].head(8).tolist()
            raise ValueError(f"{path}: infinite {column}; source indices: {rows}")
        frame[column] = values
    # Validate BEFORE astype(int): never turn 0.9/1.9/2.9 into legal labels.
    bad_label = frame["label"].notna() & ~frame["label"].isin(CLASS_LABELS)
    if bad_label.any():
        bad = frame.loc[bad_label, ["source_index", "label"]].head(8).to_dict("records")
        raise ValueError(f"{path}: label must be 0, 1 or 2; records: {bad}")
    dropped = frame.loc[frame[["delta_i", "label"]].isna().any(axis=1), "source_index"].tolist()
    if dropped:
        message = f"{path}: {len(dropped)} records have missing delta_i or label."
        if not DROP_MISSING_ROWS:
            raise ValueError(message)
        warnings.warn(message + " Dropped; see data_cleaning_report.json.", stacklevel=2)
    frame = frame.dropna(subset=["delta_i", "label"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path}: no valid records remain.")
    frame["label"] = frame["label"].astype(int)
    frame["delta_i"] = frame["delta_i"].astype(float)
    audit = {
        "path": str(path.resolve()), "input_rows": len(raw), "retained_rows": len(frame),
        "dropped_rows": len(dropped), "dropped_source_indices": dropped,
        "source_index_definition": "zero-based data-record position before cleaning",
        "class_counts": {str(c): int((frame["label"] == c).sum()) for c in CLASS_LABELS},
    }
    return frame, audit


def fit_normalization(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Normalization needs at least two finite one-dimensional training values.")
    with np.errstate(over="ignore", invalid="ignore"):
        mu, sigma = float(values.mean()), float(values.std(ddof=1))
    if not np.isfinite([mu, sigma]).all() or sigma <= 0:
        raise ValueError("Invalid normalization: mean must be finite and sample std must be positive.")
    return mu, sigma


def transform_feature(values: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("delta_i must be a finite one-dimensional array.")
    if not np.isfinite([mu, sigma]).all() or sigma <= 0:
        raise ValueError("Invalid normalization parameters.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed = (values - mu) / sigma
    if not np.isfinite(transformed).all():
        raise ValueError("Normalization produced a non-finite feature; check input magnitude.")
    return transformed.reshape(-1, 1)


def build_models() -> dict:
    return {
        "multinomial_logistic": LogisticRegression(
            C=2.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=SEED),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1,
            max_features=1.0, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_split=4, min_samples_leaf=2,
            max_features=1.0, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        "knn_5": KNeighborsClassifier(n_neighbors=5, weights="distance", p=2),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            min_samples_split=2, min_samples_leaf=1, random_state=SEED),
    }


def validate_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or not np.isin(y, CLASS_LABELS).all():
        raise ValueError("Labels must be a nonempty one-dimensional array containing only 0/1/2.")
    return y.astype(int)


def validate_proba(probability: np.ndarray, n: int) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    if probability.shape != (n, len(CLASS_LABELS)):
        raise ValueError(f"Expected probability shape {(n, len(CLASS_LABELS))}, got {probability.shape}.")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Predicted probabilities must be finite and in [0, 1].")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Predicted probabilities must sum to 1 for each sample.")
    return probability


def predict_aligned(model, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = validate_labels(model.classes_).tolist()
    if sorted(classes) != CLASS_LABELS:
        raise ValueError(f"The fitted model is missing a training class: {classes}.")
    raw = model.predict_proba(features)
    probability = validate_proba(raw[:, [classes.index(c) for c in CLASS_LABELS]], len(features))
    prediction = validate_labels(model.predict(features))
    return prediction, probability


def calculate_metrics(y_true, y_pred) -> tuple[dict, pd.DataFrame, np.ndarray]:
    y_true, y_pred = validate_labels(y_true), validate_labels(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("True and predicted labels must have the same length.")
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    per_f1 = f1_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    rows = []
    for i, cls in enumerate(CLASS_LABELS):
        tp = int(cm[i, i])
        fn, fp = int(cm[i, :].sum() - tp), int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        rows.append({
            "class_label": cls, "class_name": CLASS_NAMES[cls], "support": tp + fn,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "f1": float(per_f1[i]),
        })
    detail = pd.DataFrame(rows)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # np.mean, not pandas.mean: undefined classes MUST NOT be silently skipped.
        "sensitivity": float(np.mean(detail["sensitivity"].to_numpy())),
        "specificity": float(np.mean(detail["specificity"].to_numpy())),
        "f1_macro": float(np.mean(per_f1)),
    }
    return metrics, detail, cm


def macro_auc(y_true, probability, require_all: bool = True) -> float:
    y_true = validate_labels(y_true)
    probability = validate_proba(probability, len(y_true))
    missing = sorted(set(CLASS_LABELS) - set(y_true.tolist()))
    if missing:
        if require_all:
            raise ValueError(f"Macro OvR AUC needs all true classes; missing {missing}.")
        return float("nan")
    value = float(roc_auc_score(y_true, probability, multi_class="ovr", average="macro", labels=CLASS_LABELS))
    if not np.isfinite(value):
        raise ValueError("Unexpected non-finite AUC despite complete class coverage.")
    return value


def cv_all_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = df["delta_i"].to_numpy(dtype=float)
    y = validate_labels(df["label"].to_numpy())
    counts = {c: int((y == c).sum()) for c in CLASS_LABELS}
    if min(counts.values()) < N_SPLITS:
        raise ValueError(f"Each class needs at least {N_SPLITS} samples for this CV; counts={counts}.")
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    prepared, normalization_rows = [], []
    fold_id, seen = np.zeros(len(y), dtype=int), np.zeros(len(y), dtype=int)
    oof_feature = np.full(len(y), np.nan)
    for fold, (train, valid) in enumerate(splitter.split(x.reshape(-1, 1), y), 1):
        if len(train) < 5:
            raise ValueError("Each training fold needs at least 5 samples for knn_5.")
        try:
            mu, sigma = fit_normalization(x[train])
            x_train = transform_feature(x[train], mu, sigma)
            x_valid = transform_feature(x[valid], mu, sigma)
        except ValueError as exc:
            raise ValueError(f"Fold {fold}: {exc}") from exc
        prepared.append((fold, train, valid, x_train, x_valid))
        normalization_rows.append({"fold": fold, "n_train": len(train), "n_valid": len(valid), "mu": mu, "sigma": sigma})
        fold_id[valid], oof_feature[valid] = fold, x_valid[:, 0]
        seen[valid] += 1
    if not np.all(seen == 1):
        raise RuntimeError("Each development sample must occur in exactly one validation fold.")
    summaries, oof_frames, fold_rows = [], [], []
    for name, prototype in build_models().items():
        print(f"\nModel: {name}")
        oof_pred = np.full(len(y), -1, dtype=int)
        oof_proba = np.full((len(y), len(CLASS_LABELS)), np.nan)
        model_folds = []
        for fold, train, valid, x_train, x_valid in prepared:
            estimator = clone(prototype).fit(x_train, y[train])
            pred, probability = predict_aligned(estimator, x_valid)
            oof_pred[valid], oof_proba[valid] = pred, probability
            metrics, _, _ = calculate_metrics(y[valid], pred)
            record = {"model": name, "fold": fold, "n_train": len(train), "n_valid": len(valid),
                      **metrics, "auc_ovr_macro": macro_auc(y[valid], probability)}
            model_folds.append(record)
            fold_rows.append(record)
            print(f"  Fold {fold}: " + ", ".join(f"{k}={metrics[k]:.5f}" for k in METRIC_KEYS))
        row = {"model": name}
        for key in METRIC_KEYS:
            values = [r[key] for r in model_folds]
            row[key], row[f"{key}_std"] = float(np.mean(values)), float(np.std(values, ddof=1))
        row["auc_ovr_macro"] = macro_auc(y, oof_proba)
        fold_auc = [r["auc_ovr_macro"] for r in model_folds]
        row["auc_fold_mean"], row["auc_fold_std"] = float(np.mean(fold_auc)), float(np.std(fold_auc, ddof=1))
        summaries.append(row)
        result = pd.DataFrame({"model": name, "sample_index": np.arange(len(y)),
                               "source_index": df["source_index"].to_numpy(), "fold": fold_id,
                               "delta_i": x, "feature": oof_feature, "true": y, "pred": oof_pred})
        result[PROBA_COLUMNS] = oof_proba
        oof_frames.append(result)
    return pd.DataFrame(summaries), pd.concat(oof_frames, ignore_index=True), pd.DataFrame(fold_rows), pd.DataFrame(normalization_rows)

def calculate_model_auc_data(oof: pd.DataFrame, model_name: str) -> dict:
    group = oof.loc[oof["model"] == model_name].sort_values("sample_index")
    if group["sample_index"].duplicated().any():
        raise ValueError(f"Duplicate OOF samples for {model_name}.")
    y = validate_labels(group["true"].to_numpy())
    score = validate_proba(group[PROBA_COLUMNS].to_numpy(), len(y))
    official = macro_auc(y, score)
    class_auc = {}
    for i, cls in enumerate(CLASS_LABELS):
        binary = (y == cls).astype(int)
        class_auc[cls] = float(roc_auc_score(binary, score[:, i]))
    if not np.isclose(official, np.mean(list(class_auc.values())), atol=1e-12, rtol=0):
        raise RuntimeError("Official Macro AUC disagrees with the mean of class AUCs.")
    return {"class_auc": class_auc, "official_macro_auc": official}


def save_roc_auc_report(oof: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for name in build_models():
        data = calculate_model_auc_data(oof, name)
        rows.append({
            "model": name,
            **{f"auc_class_{c}": data["class_auc"][c] for c in CLASS_LABELS},
            "auc_ovr_macro": data["official_macro_auc"],
        })
    details = pd.DataFrame(rows)
    save_csv(details, outdir / "roc_auc_by_model_and_class.csv")
    return details


# 6. OOF diagnostics and final model / external validation.
def save_oof_diagnostics(oof: pd.DataFrame, outdir: Path) -> None:
    pooled_rows, class_frames, cm_rows = [], [], []
    for name, group in oof.groupby("model", sort=False):
        metrics, detail, cm = calculate_metrics(group["true"], group["pred"])
        pooled_rows.append({"model": name, **metrics, "auc_ovr_macro": macro_auc(group["true"], group[PROBA_COLUMNS])})
        class_frames.append(detail.assign(model=name))
        print(f"\n{name} OOF confusion matrix (rows=true, columns=predicted):\n{cm}")
        for i, true_cls in enumerate(CLASS_LABELS):
            for j, pred_cls in enumerate(CLASS_LABELS):
                cm_rows.append({"model": name, "true_label": true_cls, "predicted_label": pred_cls, "count": int(cm[i, j])})
    save_csv(pd.DataFrame(pooled_rows), outdir / "oof_metrics.csv")
    save_csv(pd.concat(class_frames, ignore_index=True), outdir / "oof_metrics_by_class.csv")
    save_csv(pd.DataFrame(cm_rows), outdir / "oof_confusion_matrices.csv")
    table = oof.pivot(index="sample_index", columns="model", values="pred")
    if table.isna().any().any():
        raise ValueError("OOF prediction comparison requires identical sample coverage across models.")
    names = list(build_models())
    agreement = pd.DataFrame([[float((table[a] == table[b]).mean()) for b in names] for a in names], index=names, columns=names)
    save_csv(agreement, outdir / "model_prediction_agreement.csv", index=True)


def fit_final_model(df: pd.DataFrame, model_name: str) -> tuple[object, float, float]:
    x = df["delta_i"].to_numpy(dtype=float)
    y = validate_labels(df["label"].to_numpy())
    if set(y.tolist()) != set(CLASS_LABELS):
        raise ValueError("Final training requires all three classes.")
    mu, sigma = fit_normalization(x)
    estimator = clone(build_models()[model_name]).fit(transform_feature(x, mu, sigma), y)
    return estimator, mu, sigma


def save_external_confusion_matrix_reports(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cm: np.ndarray,
    model_name: str,
    outdir: Path,
) -> None:

    y_true = validate_labels(y_true)
    y_pred = validate_labels(y_pred)
    cm = np.asarray(cm, dtype=int)
    expected_shape = (len(CLASS_LABELS), len(CLASS_LABELS))
    if cm.shape != expected_shape:
        raise ValueError(f"Expected confusion matrix shape {expected_shape}, got {cm.shape}.")
    if cm.sum() != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("Confusion matrix counts do not match external sample count.")

    true_names = [f"true_{c}_{CLASS_NAMES[c]}" for c in CLASS_LABELS]
    pred_names = [f"pred_{c}_{CLASS_NAMES[c]}" for c in CLASS_LABELS]

    # 1. Standard count matrix.
    matrix = pd.DataFrame(cm, index=true_names, columns=pred_names)
    save_csv(matrix, outdir / "external_confusion_matrix.csv", index=True)

    # 2. Long/tidy cell-level representation.
    long_rows = []
    for i, true_cls in enumerate(CLASS_LABELS):
        row_total = int(cm[i, :].sum())
        for j, pred_cls in enumerate(CLASS_LABELS):
            count = int(cm[i, j])
            long_rows.append({
                "model": model_name,
                "true_label": true_cls,
                "true_class_name": CLASS_NAMES[true_cls],
                "predicted_label": pred_cls,
                "predicted_class_name": CLASS_NAMES[pred_cls],
                "count": count,
                "true_class_total": row_total,
                "row_percentage": count / row_total if row_total else np.nan,
            })
    save_csv(pd.DataFrame(long_rows), outdir / "external_confusion_matrix_long.csv")

    row_totals = cm.sum(axis=1, keepdims=True).astype(float)
    normalized = np.divide(
        cm.astype(float),
        row_totals,
        out=np.full(cm.shape, np.nan, dtype=float),
        where=row_totals != 0,
    )
    normalized_matrix = pd.DataFrame(normalized, index=true_names, columns=pred_names)
    save_csv(
        normalized_matrix,
        outdir / "external_confusion_matrix_normalized_true.csv",
        index=True,
    )

    # 4. One-vs-rest confusion-matrix statistics for each class.
    total = int(cm.sum())
    by_class_rows = []
    for i, cls in enumerate(CLASS_LABELS):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)

        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        precision = tp / (tp + fp) if tp + fp else np.nan
        npv = tn / (tn + fn) if tn + fn else np.nan
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if np.isfinite(precision) and np.isfinite(sensitivity)
            and (precision + sensitivity) > 0
            else 0.0
        )

        by_class_rows.append({
            "model": model_name,
            "class_label": cls,
            "class_name": CLASS_NAMES[cls],
            "support": tp + fn,
            "predicted_count": tp + fp,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "sensitivity_recall": sensitivity,
            "specificity": specificity,
            "precision_ppv": precision,
            "npv": npv,
            "f1": f1,
            "one_vs_rest_accuracy": (tp + tn) / total if total else np.nan,
        })

    save_csv(
        pd.DataFrame(by_class_rows),
        outdir / "external_confusion_matrix_by_class.csv",
    )


def evaluate_external(df: pd.DataFrame, estimator, mu: float, sigma: float,
                      model_name: str, outdir: Path) -> pd.DataFrame:
    y = validate_labels(df["label"].to_numpy())
    x = transform_feature(df["delta_i"].to_numpy(dtype=float), mu, sigma)
    pred, probability = predict_aligned(estimator, x)
    metrics, per_class, cm = calculate_metrics(y, pred)
    missing = sorted(set(CLASS_LABELS) - set(y.tolist()))
    status = "complete" if not missing else "incomplete_true_class_coverage"
    auc_status = "ok" if not missing else "undefined_missing_true_classes"
    if missing:
        warnings.warn(f"External data missing classes {missing}: three-class Macro AUC is undefined; see external_metrics.csv.", stacklevel=2)
    summary = pd.DataFrame([{"model": model_name, "n_samples": len(y), "class_coverage_status": status,
                             "missing_classes": ",".join(map(str, missing)), **metrics,
                             "auc_ovr_macro": macro_auc(y, probability, require_all=False), "auc_status": auc_status}])
    for i, cls in enumerate(CLASS_LABELS):
        binary = (y == cls).astype(int)
        per_class.loc[i, "auc_ovr"] = float(roc_auc_score(binary, probability[:, i])) if 0 < binary.sum() < len(y) else np.nan
    detail = df.copy()
    detail["feature"], detail["predicted"] = x[:, 0], pred
    detail[PROBA_COLUMNS] = probability
    save_csv(detail, outdir / "external_predictions.csv")
    save_csv(summary, outdir / "external_metrics.csv")
    save_csv(per_class.assign(model=model_name), outdir / "external_metrics_by_class.csv")


    save_external_confusion_matrix_reports(y, pred, cm, model_name, outdir)

    print(
        "External confusion matrix (rows=true, columns=predicted):\n"
        + pd.DataFrame(
            cm,
            index=[f"true_{c}_{CLASS_NAMES[c]}" for c in CLASS_LABELS],
            columns=[f"pred_{c}_{CLASS_NAMES[c]}" for c in CLASS_LABELS],
        ).to_string()
    )
    return summary


# 7. Entry point.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev", type=Path, default=Path(DEV_CSV), help="Development CSV; required to exist.")
    parser.add_argument("--external", type=Path, default=Path(EXT_CSV), help="Optional external CSV.")
    parser.add_argument("--outdir", type=Path, default=OUTDIR, help="Output directory; use a new directory for each run.")
    parser.add_argument("--skip-external", action="store_true", help="Do not read or evaluate external data.")
    args = parser.parse_args(argv)
    validate_config()
    if not args.skip_external and args.external.exists() and args.dev.resolve() == args.external.resolve():
        raise ValueError("Development and external validation must not be the same file.")
    df, dev_audit = load(args.dev)
    ext, ext_audit, external_status = None, None, "skipped_by_request"
    if not args.skip_external:
        if args.external.exists():
            ext, ext_audit = load(args.external)
            external_status = "loaded"
        else:
            external_status = "file_not_found"
            print(f"[Notice] External CSV not found: {args.external}; external validation will be skipped.")
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    save_json({"development": dev_audit, "external": ext_audit}, outdir / "data_cleaning_report.json")
    print(f"Development samples: {len(df)}; class counts: {dev_audit['class_counts']}")
    results, oof, folds, normalization = cv_all_models(df)
    save_csv(results, outdir / "cv_metrics.csv")
    save_csv(folds, outdir / "cv_fold_metrics.csv")
    save_csv(normalization, outdir / "cv_fold_normalization_params.csv")
    save_csv(oof, outdir / "oof_predictions.csv")
    table = results.set_index("model")[METRIC_KEYS + ["auc_ovr_macro"]].T
    table.index = ["Accuracy", "Sensitivity", "Specificity", "F1", "AUC"]
    save_csv(table, outdir / "cv_metrics_transposed.csv", index=True)
    print("\nCV: first four metrics are fold means; AUC is pooled OOF Macro OvR.\n" + table.round(5).to_string())
    ranking = results[["model", "auc_ovr_macro"]].sort_values("auc_ovr_macro", ascending=False, kind="mergesort").reset_index(drop=True)
    save_csv(ranking, outdir / "auc_results.csv")
    best = str(ranking.iloc[0]["model"])
    print(f"\nHighest OOF AUC: {best}; exact ties use the configured model order.")
    save_oof_diagnostics(oof, outdir)
    save_roc_auc_report(oof, outdir)
    estimator, mu, sigma = fit_final_model(df, best)
    save_json({"mu": mu, "sigma": sigma, "ddof": 1, "formula": "z = (delta_i - mu) / sigma",
               "scope": "Full development data; final model and external validation ONLY.",
               "cv_parameters": "cv_fold_normalization_params.csv", "selected_model": best},
              outdir / "normalization_params.json")
    joblib.dump({"estimator": estimator, "model_name": best, "mu": mu, "sigma": sigma,
                 "feature_column": "delta_i", "classes": CLASS_LABELS,
                 "sklearn_version": sklearn.__version__}, outdir / "best_model.joblib")
    if ext is not None:
        evaluate_external(ext, estimator, mu, sigma, best, outdir)
        external_status = "evaluated"
    source = Path(__file__)
    save_json({
        "status": "completed", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selected_model": best, "selection_criterion": "pooled OOF macro OvR AUC; external data not used for selection",
        "external_status": external_status,
        "class_labels": CLASS_LABELS, "class_names": CLASS_NAMES, "seed": SEED, "n_splits": N_SPLITS,
        "normalization_ddof": 1,
        "metric_notes": {
            "cv_metrics": "First four metrics are unweighted fold means; *_std uses ddof=1 and is not a confidence interval.",
            "auc_ovr_macro": "Calculated from pooled OOF probabilities, not a mean of fold AUCs.",
            "auc_fold_mean": "Separate mean of validation-fold AUCs; not used for selection.",
            "oof_metrics": "All classification metrics recomputed on pooled OOF predictions.",
            "sensitivity_specificity": "Fixed-class macro mean; undefined denominator propagates NaN.",
            "f1_macro": "Fixed labels=[0,1,2]; zero_division=0, including absent classes.",
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }, outdir / "run_metadata.json")
    print(f"\nComplete. Output directory: {outdir.resolve()}")


if __name__ == "__main__":
    main()

