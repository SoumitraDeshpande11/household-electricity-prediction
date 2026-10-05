"""Build the reproducible household electricity consumption dataset.

The case study asks for household-level attributes that are not present in the
public UCI/Kaggle minute-level household power file.  This script therefore
generates a synthetic panel of households, while using the UCI file as an
optional calibration source when it is available locally.  The generated
records are intentionally deterministic for a given seed and contain a small,
known amount of missing meter data for the notebook's imputation exercise.

Run from the repository root with::

    python3 scripts/build_dataset.py

An optional UCI/Kaggle download can be supplied with ``--reference-path``.  A
missing or unreadable reference file does not stop the build; the documented
UCI monthly scale is used as the fallback calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "household_monthly_panel.csv"
DEFAULT_METADATA = ROOT / "data" / "processed" / "dataset_metadata.json"

DEFAULT_SEED = 42
DEFAULT_HOUSEHOLDS = 500
DEFAULT_MONTHS = 36
# A typical UCI household's monthly active-energy usage is in this broad
# range.  It is used only as a scale anchor for the synthetic generator.
FALLBACK_REFERENCE_MONTHLY_KWH = 650.0

SEASON_BY_MONTH = {
    1: "Winter",
    2: "Winter",
    3: "Summer",
    4: "Summer",
    5: "Summer",
    6: "Summer",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Autumn",
    11: "Autumn",
    12: "Winter",
}

SOURCE_URLS = {
    "kaggle": "https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set",
    "uci": "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption",
}


def _read_reference_monthly_kwh(reference_path: Path) -> tuple[float | None, str]:
    """Estimate the monthly kWh median from a local UCI/Kaggle file.

    The UCI download is a semicolon-separated text file with a
    ``Global_active_power`` column.  Some Kaggle copies are CSV files, so the
    parser first tries semicolon and then comma delimiters.  The function is
    deliberately best-effort: dataset generation remains available offline.
    """

    if not reference_path.exists():
        return None, f"reference file not found: {reference_path}"

    last_error = "unknown parsing error"
    for separator in (";", ",", "\t"):
        try:
            header = pd.read_csv(reference_path, sep=separator, nrows=0)
            columns = {str(column).strip(): column for column in header.columns}
            power_column = columns.get("Global_active_power")
            date_column = columns.get("Date")
            if power_column is None or date_column is None:
                continue

            chunks: list[pd.DataFrame] = []
            for chunk in pd.read_csv(
                reference_path,
                sep=separator,
                usecols=[date_column, power_column],
                na_values=["?", "", "NA", "NaN"],
                chunksize=250_000,
            ):
                dates = pd.to_datetime(chunk[date_column], dayfirst=True, errors="coerce")
                power = pd.to_numeric(chunk[power_column], errors="coerce")
                valid = dates.notna() & power.notna()
                if valid.any():
                    chunks.append(
                        pd.DataFrame(
                            {
                                "month": dates.loc[valid].dt.to_period("M").astype(str),
                                # Global active power is kW per one-minute row.
                                "kwh": power.loc[valid].astype(float) / 60.0,
                            }
                        )
                    )
            if not chunks:
                return None, "reference file contained no valid power readings"

            readings = pd.concat(chunks, ignore_index=True)
            monthly = readings.groupby("month", sort=True)["kwh"].sum()
            median = float(monthly.median())
            if np.isfinite(median) and median > 0:
                return median, f"calibrated from {reference_path} using {separator!r} delimiter"
            return None, "reference file produced a non-positive monthly median"
        except (OSError, ValueError, TypeError, pd.errors.ParserError) as exc:
            last_error = str(exc)

    return None, f"reference file could not be parsed: {last_error}"


def _generate_panel(
    *,
    n_households: int,
    n_months: int,
    seed: int,
    reference_monthly_kwh: float,
) -> pd.DataFrame:
    """Generate a household-month panel with realistic seasonal dependence."""

    if n_households < 1 or n_months < 2:
        raise ValueError("n_households must be >= 1 and n_months must be >= 2")

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_months, freq="MS")
    season_names = np.array([SEASON_BY_MONTH[int(month)] for month in dates.month])

    # Household characteristics remain stable across the panel.  The mild
    # correlations make the synthetic records more plausible than independent
    # random columns (larger households tend to have more rooms/appliances).
    household_size = rng.integers(1, 8, size=n_households)
    number_of_rooms = np.clip(
        household_size + rng.integers(-1, 4, size=n_households), 1, 10
    )
    appliance_count = np.clip(
        2 + household_size * 2 + rng.poisson(3, size=n_households), 3, 28
    )
    ac_propensity = np.clip(rng.beta(2.2, 4.0, size=n_households), 0.02, 0.98)
    efficiency_factor = np.clip(rng.normal(1.0, 0.08, size=n_households), 0.78, 1.24)

    # Month-specific effects model the load shape.  AC hours are both a feature
    # and a major source of summer demand, while the remaining seasonal effect
    # captures lighting and cooling/heating demand not represented by AC.
    ac_hours_by_season = {"Winter": 18.0, "Summer": 190.0, "Monsoon": 95.0, "Autumn": 48.0}
    non_ac_effect = {"Winter": -28.0, "Summer": 108.0, "Monsoon": 42.0, "Autumn": 8.0}

    records: list[dict[str, Any]] = []
    # Use a slowly changing trend to avoid making month itself irrelevant.
    trend = np.linspace(0.0, 0.035, n_months)
    reference_scale = float(reference_monthly_kwh) / FALLBACK_REFERENCE_MONTHLY_KWH

    for household_index in range(n_households):
        size = int(household_size[household_index])
        rooms = int(number_of_rooms[household_index])
        appliances = int(appliance_count[household_index])
        ac_affinity = float(ac_propensity[household_index])
        efficiency = float(efficiency_factor[household_index])

        # A warm start gives the first row a meaningful lag, while subsequent
        # rows use the previous generated monthly consumption exactly.
        previous = float(
            (55 + 33 * size + 11 * rooms + 6 * appliances)
            * efficiency
            * (0.92 + 0.08 * rng.random())
        )

        for month_index, date in enumerate(dates):
            season = str(season_names[month_index])
            season_ac = ac_hours_by_season[season]
            ac_hours = max(
                0.0,
                season_ac * ac_affinity * (0.88 + 0.025 * size)
                + rng.normal(0.0, 9.0 if season != "Summer" else 16.0),
            )

            # Structural demand is intentionally additive and easy to explain
            # in the notebook.  The lag coefficient creates temporal continuity.
            structural = (
                55.0
                + 33.0 * size
                + 11.0 * rooms
                + 6.0 * appliances
                + 1.75 * ac_hours
                + non_ac_effect[season]
            )
            noise_sd = 18.0 + 0.035 * structural
            current = (
                0.63 * structural
                + 0.31 * previous
                + rng.normal(0.0, noise_sd)
            )
            current *= efficiency * (1.0 + trend[month_index]) * reference_scale
            current = max(35.0, float(current))

            records.append(
                {
                    "household_id": f"HH{household_index + 1:04d}",
                    "month": date.strftime("%Y-%m-%d"),
                    "season": season,
                    "household_size": size,
                    "number_of_rooms": rooms,
                    "appliance_count": appliances,
                    "ac_usage_hours": round(ac_hours, 2),
                    "previous_month_consumption": round(previous, 2),
                    "monthly_consumption": round(current, 2),
                }
            )
            previous = current

    data = pd.DataFrame.from_records(records)
    # Missingness is injected after the target is generated so that the target
    # remains complete and the notebook can demonstrate feature imputation.
    missing_rates = {
        "household_size": 0.012,
        "number_of_rooms": 0.012,
        "appliance_count": 0.018,
        "ac_usage_hours": 0.030,
        "previous_month_consumption": 0.025,
    }
    missing_counts: dict[str, int] = {}
    for column, rate in missing_rates.items():
        mask = rng.random(len(data)) < rate
        # The dataset is large enough that these are practically guaranteed,
        # but this keeps the guarantee explicit for custom tiny builds.
        if not mask.any():
            mask[int(rng.integers(0, len(data)))] = True
        data.loc[mask, column] = np.nan
        missing_counts[column] = int(mask.sum())

    # Keep a stable, documented column order and a parseable date column.
    ordered_columns = [
        "household_id",
        "month",
        "season",
        "household_size",
        "number_of_rooms",
        "appliance_count",
        "ac_usage_hours",
        "previous_month_consumption",
        "monthly_consumption",
    ]
    data = data[ordered_columns]
    data.attrs["missing_counts"] = missing_counts
    return data


def build_dataset(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    metadata_path: Path = DEFAULT_METADATA,
    n_households: int = DEFAULT_HOUSEHOLDS,
    n_months: int = DEFAULT_MONTHS,
    seed: int = DEFAULT_SEED,
    reference_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate and persist the panel, returning the data and metadata."""

    reference_value = FALLBACK_REFERENCE_MONTHLY_KWH
    reference_note = "fallback calibration based on the UCI dataset's monthly scale"
    reference_mode = "fallback"
    if reference_path is not None:
        calibrated, note = _read_reference_monthly_kwh(reference_path)
        if calibrated is not None:
            reference_value = calibrated
            reference_note = note
            reference_mode = "local_reference_file"
        else:
            reference_note = f"{note}; {reference_note}"

    data = _generate_panel(
        n_households=n_households,
        n_months=n_months,
        seed=seed,
        reference_monthly_kwh=reference_value,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, float_format="%.2f")

    data_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata: dict[str, Any] = {
        "dataset_name": "Household Electricity Consumption Prediction",
        "synthetic": True,
        "rows": int(len(data)),
        "households": int(n_households),
        "months_per_household": int(n_months),
        "date_range": [str(data["month"].min()), str(data["month"].max())],
        "seed": int(seed),
        "reference_calibration_kwh_median": round(float(reference_value), 4),
        "reference_calibration_mode": reference_mode,
        "reference_calibration_note": reference_note,
        "source_urls": SOURCE_URLS,
        "columns": list(data.columns),
        "missing_counts": {
            column: int(count) for column, count in data.isna().sum().items() if count
        },
        "target_missing_count": int(data["monthly_consumption"].isna().sum()),
        "sha256": data_hash,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return data, metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--households", type=int, default=DEFAULT_HOUSEHOLDS)
    parser.add_argument("--months", type=int, default=DEFAULT_MONTHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
        help="optional local UCI/Kaggle household power file for monthly-scale calibration",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        data, metadata = build_dataset(
            output_path=args.output,
            metadata_path=args.metadata,
            n_households=args.households,
            n_months=args.months,
            seed=args.seed,
            reference_path=args.reference_path,
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"Dataset build failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(data):,} rows to {args.output}")
    print(f"Wrote metadata to {args.metadata}")
    print(f"Columns: {', '.join(data.columns)}")
    print(f"Missing values: {metadata['missing_counts']}")
    print(f"SHA-256: {metadata['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
