"""
Feature drift and score distribution monitoring.

Uses:
- KS test for continuous feature drift
- PSI (Population Stability Index) for score drift
- Chi-squared for categorical drift
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger
from src.utils.paths import ARTIFACTS_METRICS, ensure_dirs

logger = get_logger(__name__)


# KS test
# -----------------------------------------------------------------------------------------


def ks_drift_test(
    reference: pd.Series,
    current: pd.Series,
    threshold: float = 0.1,
) -> dict:
    """
    Kolmogorov-Smirnov test for distribution shift.

    Returns dict with: statistic, p_value, drifted (bool), threshold.
    """
    ks_stat, p_value = stats.ks_2samp(reference.dropna(), current.dropna())
    return {
        "statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_value), 4),
        "drifted": bool(ks_stat > threshold),
        "threshold": threshold,
    }


# PSI
# -----------------------------------------------------------------------------------------


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Population Stability Index.
    PSI < 0.1  -> no shift
    PSI 0.1-0.2 -> minor shift
    PSI > 0.2  -> significant shift
    """
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    ref_pct = ref_counts / len(reference) + eps
    cur_pct = cur_counts / len(current) + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 4)


# Full drift report
# -----------------------------------------------------------------------------------------


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_cols: Optional[list[str]] = None,
    ks_threshold: float = 0.1,
    psi_threshold: float = 0.2,
    save: bool = True,
) -> dict:
    """
    Run drift checks across all specified columns.

    Args:
        reference_df: Training data (baseline).
        current_df: New batch of scoring data.
        numeric_cols: Columns to test. If None, uses all shared numeric cols.
        ks_threshold: KS statistic threshold for flagging drift.
        psi_threshold: PSI threshold for flagging drift.
        save: If True, save report to artifacts/metrics/drift_report.json.

    Returns:
        Dict with per-feature drift results and summary.
    """
    if numeric_cols is None:
        numeric_cols = list(
            set(reference_df.select_dtypes(include=[np.number]).columns)
            & set(current_df.select_dtypes(include=[np.number]).columns)
        )

    results = {}
    drifted_features = []

    for col in numeric_cols:
        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values
        if len(ref_vals) < 20 or len(cur_vals) < 20:
            continue

        ks = ks_drift_test(pd.Series(ref_vals), pd.Series(cur_vals), threshold=ks_threshold)
        psi_val = compute_psi(ref_vals, cur_vals)

        results[col] = {
            "ks_statistic": ks["statistic"],
            "ks_p_value": ks["p_value"],
            "ks_drifted": ks["drifted"],
            "psi": psi_val,
            "psi_drifted": psi_val > psi_threshold,
        }

        if ks["drifted"] or psi_val > psi_threshold:
            drifted_features.append(col)
            logger.warning(
                "Drift detected in '%s': KS=%.4f, PSI=%.4f", col, ks["statistic"], psi_val
            )

    report = {
        "n_features_checked": len(results),
        "n_drifted": len(drifted_features),
        "drifted_features": drifted_features,
        "drift_rate": round(len(drifted_features) / max(len(results), 1), 3),
        "feature_results": results,
    }

    logger.info("Drift report: %d/%d features drifted.", len(drifted_features), len(results))

    if save:
        ensure_dirs()
        path = ARTIFACTS_METRICS / "drift_report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Drift report saved -> %s", path)

    return report


def check_score_distribution(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    psi_threshold: float = 0.2,
) -> dict:
    """Check PSI of the output score distribution."""
    psi_val = compute_psi(reference_scores, current_scores)
    shifted = psi_val > psi_threshold
    result = {
        "score_psi": psi_val,
        "score_shifted": shifted,
        "threshold": psi_threshold,
    }
    if shifted:
        logger.warning("Score distribution shift detected! PSI=%.4f", psi_val)
    return result
