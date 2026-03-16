"""
Model training: baseline (Logistic Regression) and final (XGBoost with
GradientBoostingClassifier fallback when XGBoost is not installed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier

    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGBOOST_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Baseline
# -----------------------------------------------------------------------------------------
def train_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict | None = None,
) -> LogisticRegression:
    """Train a Logistic Regression baseline."""
    default_params = dict(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    if params:
        default_params.update(params)

    model = LogisticRegression(**default_params)
    model.fit(X_train, y_train)
    logger.info(
        "Baseline (LR) trained. Coeff range: [%.4f, %.4f]", model.coef_.min(), model.coef_.max()
    )
    return model


# Final model
# -----------------------------------------------------------------------------------------


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    params: dict | None = None,
):
    """
    Train an XGBoost classifier. Falls back to sklearn GradientBoostingClassifier
    when XGBoost is not installed.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    default_spw = float(n_neg / max(n_pos, 1))

    if _XGBOOST_AVAILABLE:
        default_params = dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=default_spw,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
        if params:
            # Drop XGB-specific keys that don't apply to sklearn
            default_params.update(params)
        model = XGBClassifier(**default_params)
        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
        model.fit(X_train, y_train, **fit_kwargs)
        logger.info("XGBoost trained. Best iteration: %s", getattr(model, "best_iteration", "N/A"))
    else:
        logger.warning("XGBoost not installed — falling back to GradientBoostingClassifier.")
        from sklearn.ensemble import GradientBoostingClassifier

        # Map overlapping param names; ignore XGB-specific ones
        gb_params = dict(
            n_estimators=min((params or {}).get("n_estimators", 200), 200),
            max_depth=(params or {}).get("max_depth", 4),
            learning_rate=(params or {}).get("learning_rate", 0.08),
            subsample=(params or {}).get("subsample", 0.8),
            random_state=42,
        )
        model = GradientBoostingClassifier(**gb_params)
        model.fit(X_train, y_train)
        logger.info("GradientBoostingClassifier trained (XGBoost fallback).")

    return model


# Optuna tuning
# -----------------------------------------------------------------------------------------


def tune_xgboost(  # pragma: no cover
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int = 50,
    timeout: int = 300,
) -> dict:
    """
    Use Optuna to search XGBoost hyperparameters.
    Returns the best params dict.
    """
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed, skipping tuning.")
        return {}

    from sklearn.metrics import roc_auc_score

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            scale_pos_weight=float(n_neg / max(n_pos, 1)),
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_params
    logger.info("Optuna best ROC-AUC=%.4f with params: %s", study.best_value, best)
    return best
