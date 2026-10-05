# Real meter data and transformations

The source is Kaggle's copy of **Individual Household Electric Power Consumption**, originally published by UCI. Download and every processing step are implemented in household_electricity_prediction.ipynb.

- Kaggle: https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set
- UCI: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- Attribution: Hebrail, G. and Berard, A. (2006), DOI 10.24432/C58K54.
- License: CC BY 4.0.
- Download: public Kaggle API, no credentials required for the bundled archive.
- Archive size: 20,357,475 bytes. Extracted file: 132,960,755 bytes.
- Archive SHA-256: 125ea23cde40b7f745d803ba32a712b41ae1900915fe4f60cc8abe9d00b79e45.
- Raw TXT SHA-256: 4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7.

## Files

The exact downloaded ZIP and manifest are in source/. The notebook extracts the TXT into ignored raw/. No randomly generated source table, households, or targets are used.

processed/daily_consumption.csv contains 1,442 calendar days and 1,410 usable energy days. It retains original observed-minute counts, repaired-minute counts, unresolved-minute counts, and the usable-energy flag.

processed/forecast_records.csv contains 957 forecast origins with next-30-day targets. Records describe **one home**, not 957 homes. Their targets overlap.

processed/dataset_metadata.json contains reproducible source and processing counts.

## Units and missing readings

The 2,075,259 rows record one minute each. Global_active_power is mean kW; daily kWh = sum of minute power / 60. Sub_metering_1/2/3 contain minute Wh; daily circuit kWh = sum / 1,000.

The source uses '?' for missing readings. Of 25,979 missing power minutes, 156 are repaired with causal forward filling limited to five minutes. The first five minutes of a longer gap can be filled, but the day remains unusable if any unresolved power reading remains. Partial days are also excluded from targets.

Every 30-day target requires 30 usable energy days. Repair counts for each target are retained. Historical input gaps are allowed and later median-imputed using each model's training partition only.

## Model inputs and target

| Column | Definition |
| --- | --- |
| forecast_date | First day of the predicted period |
| previous_30_day_kwh | Total energy over the preceding 30 days |
| previous_7_day_mean_kwh | Mean daily energy over the preceding seven days |
| previous_day_kwh | Energy on the immediately preceding day |
| kitchen_7_day_mean_kwh | Seven-day mean kitchen-circuit energy |
| laundry_7_day_mean_kwh | Seven-day mean laundry-circuit energy |
| heating_ac_7_day_mean_kwh | Seven-day mean combined water-heater and AC circuit energy |
| month_sin, month_cos | Cyclic month encoding, derived from forecast_date |
| season | Winter, Spring, Summer, or Autumn in France |
| next_30_day_kwh | Actual energy from forecast_date through forecast_date + 29 days |
| target_repaired_minutes | Number of short-gap repaired power minutes within that target |
| forecast_end | Last day contributing to the target |

Each historical input ends before forecast_date. Calendar season is based on the origin month, not an entire horizon spanning seasons. The forecast is a fixed 30-day horizon and does not claim to match a calendar billing month.

Household size, rooms, appliance count, and separate AC hours are **absent**. Sub_metering_3 combines water heating and air conditioning; it cannot isolate AC use.

## Evaluation boundary

The latest approximately 20% of valid origins form the holdout. Training origins whose target reaches the holdout start are discarded. Five expanding training folds use a 30-row gap; assertions check target date boundaries. Validation origins may still overlap one another, so their errors are correlated.

The one-home source cannot support claims about performance on new homes. Reproduce all transformations by running the notebook from a fresh kernel.
