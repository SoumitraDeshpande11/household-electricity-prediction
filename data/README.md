# Household electricity dataset

`processed/household_monthly_panel.csv` is a deterministic synthetic panel made
for the Household Electricity Consumption Prediction case study. It contains
500 households observed monthly for 36 months (18,000 household-month rows).
The data builder is [scripts/build_dataset.py](../scripts/build_dataset.py).

## Source and lineage

The public source used as the modelling inspiration is the [Individual
Household Electric Power Consumption dataset at UCI](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption),
also distributed through this [Kaggle mirror](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set).
That source contains minute-level readings such as `Global_active_power`,
voltage, and sub-metering values for one household. It does not contain the
case-study attributes household size, room count, appliance count, or AC
usage. Consequently, the repository does not pretend that those fields were
observed in the source: the processed file is a synthetic, household-level
panel calibrated to the broad monthly kWh scale of the UCI measurements.

If a local UCI/Kaggle file is available, it can be used only for the scale
calibration step:

```bash
python3 scripts/build_dataset.py \
  --reference-path data/raw/household_power_consumption.txt
```

With no local reference file (the default and the current build), the script
uses a documented fallback monthly scale of 650 kWh. No network download is
required, so the build is reproducible in a clean environment.

## Generation method

The fixed seed is 42. Each household receives a stable household size, room
count, appliance count, AC-use propensity, and efficiency factor. Monthly AC
hours are higher in summer and lower in winter. Consumption combines those
attributes with a seasonal effect, a small time trend, Gaussian meter noise,
and 31% of the previous month's consumption. The first month's lag is a
warm-start estimate because a 36-month extract has no earlier row.

After the complete target is generated, reproducible missing meter readings
are injected into the numeric feature columns: household size (1.2%), rooms
(1.2%), appliances (1.8%), AC hours (3.0%), and previous consumption (2.5%).
The target is never made missing. These values are intentional so the notebook
can show missing-value detection and imputation without hiding a target leak.

## Columns

| Column | Meaning |
| --- | --- |
| `household_id` | Stable synthetic household identifier |
| `month` | Month start date (`YYYY-MM-DD`) |
| `season` | Winter, Summer, Monsoon, or Autumn |
| `household_size` | Number of people in the household |
| `number_of_rooms` | Number of rooms |
| `appliance_count` | Count of regularly used appliances |
| `ac_usage_hours` | Monthly air-conditioner hours |
| `previous_month_consumption` | Previous month's generated consumption |
| `monthly_consumption` | Target monthly electricity consumption in kWh-equivalent units |

## Rebuilding and checking

From the repository root:

```bash
python3 scripts/build_dataset.py
```

The command writes the CSV and `processed/dataset_metadata.json`, including
the seed, missing-value counts, source URLs, and SHA-256 digest. Changing the
seed intentionally changes the generated panel; leaving the defaults produces
the committed exam dataset with 18,000 rows.
