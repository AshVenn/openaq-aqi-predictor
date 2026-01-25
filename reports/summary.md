# Air Quality Modeling Summary

## Dataset
- Source: World Air Quality – OpenAQ (local CSV)
- Rows after cleaning: {{rows_cleaned}}
- Time range: {{time_start}} to {{time_end}}
- Aggregation window: {{aggregation_window}}

## AQI
- AQI computed using US EPA breakpoints.
- % Missing AQI: {{pct_missing_aqi}}

## Model
- Target: Numeric AQI (regression)
- Baseline: Linear Regression
- Best model: {{best_model_name}}

## Metrics (test set)
- MAE: {{mae}}
- RMSE: {{rmse}}
- R2: {{r2}}

## Notes
- {{notes}}
