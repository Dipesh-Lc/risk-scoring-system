"""
Batch scoring pipeline: score a CSV of customer records and save results.

Run:
    python -m src.pipelines.batch_scoring_pipeline \
        --input data/samples/sample_customers.csv \
        --output artifacts/scores/batch_results.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import ARTIFACTS_SCORES, DATA_SAMPLES, ensure_dirs

logger = get_logger(__name__)


def run(input_path: Path, output_path: Path | None = None) -> pd.DataFrame:
    """
    Load a CSV of customer records, validate, score, and save results.

    Args:
        input_path: Path to input CSV (raw UCI feature columns).
        output_path: Where to write scored results. Defaults to artifacts/scores/.

    Returns:
        DataFrame with original columns + default_probability, risk_score, risk_band.
    """
    ensure_dirs()
    start = time.time()

    # Load
    # ---------------------------------------------------------------------------------
    logger.info("Loading batch input: %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d records x %d columns.", *df.shape)

    # Validate
    # ---------------------------------------------------------------------------------
    from src.monitoring.data_quality import check_batch

    quality = check_batch(df)
    if not quality.valid:
        for err in quality.errors:
            logger.error("Batch quality error: %s", err)
        raise ValueError(f"Batch input failed quality checks: {quality.errors}")
    for warn in quality.warnings:
        logger.warning("Batch quality warning: %s", warn)

    # Load artifacts
    # ---------------------------------------------------------------------------------
    from src.models.registry import load_artifact

    model = load_artifact("xgb_calibrated")
    scaler = load_artifact("scaler")
    feature_names = load_artifact("feature_names")

    # Score
    # ---------------------------------------------------------------------------------
    from src.models.predict import predict_batch

    scored = predict_batch(df, model=model, scaler=scaler, feature_names=feature_names)

    # Summary
    # ---------------------------------------------------------------------------------
    band_counts = scored["risk_band"].value_counts().to_dict()
    avg_prob = scored["default_probability"].mean()
    logger.info("Scoring summary:")
    logger.info("  Average default probability: %.2f%%", avg_prob * 100)
    for band in ["Low", "Medium", "High"]:
        count = band_counts.get(band, 0)
        pct = count / len(scored) * 100
        logger.info("  %s: %d (%.1f%%)", band, count, pct)

    # Save
    # ---------------------------------------------------------------------------------
    if output_path is None:
        output_path = ARTIFACTS_SCORES / "batch_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)

    elapsed = time.time() - start
    logger.info(
        "Batch scoring complete: %d records in %.2fs -> %s", len(scored), elapsed, output_path
    )

    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch scoring pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_SAMPLES / "sample_customers.csv",
        help="Path to input CSV",
    )
    parser.add_argument("--output", type=Path, default=None, help="Path for scored output CSV")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        raise SystemExit(1)

    run(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
