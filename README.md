# Macro Engine V1

# V1 Macro Engine - Summary and Improvement Blueprint

## 1. What v1 is?
- **Type:** Macro forecasting engine.
- **Purpose:** Generate 20-year quarterly macro scenarios as inputs for downstream PD/LGD workflows.
- **Status:** Runnable and useful as a **prototype baseline**.

## 2. V1 Architecture (3 Phases)
1. **Q1-Q8 (1-2 years):** Statistical forecast via VAR/BVAR.
2. **Q9-Q20 (2-5 years):** Bridge from short-run model path toward structural long-run anchors.
3. **Q21-Q80 (5-20 years):** Scenario overlays (Baseline, Mild Adverse, Severe Adverse, Demographic Low Growth).

## 3. What V1 Produces
- Full scenario table: `outputs/macro_forecast_paths.csv`
- Short-horizon intervals: `outputs/macro_forecast_short_horizon_intervals.csv`
- Diagnostics: `outputs/macro_model_diagnostics.json`
- Anchor assumptions snapshot: `outputs/macro_anchor_assumptions.json`
- PD sample export: `outputs/pd_macro_subset_sample.csv` + `.json`

## 4. Key V1 Files
- Data build: `scripts/fetch_macro_panel_fred.py`
- Forecast engine: `scripts/run_macro_forecast_engine.py`
- PD export: `scripts/export_pd_macro_subset.py`
- Backtest (manual, not in main pipeline): `scripts/backtest_bvar_oos.py`
- Config: `macro_engine_config.json`
- One-command run: `run_all.sh`

## 5. Run V1
```bash
./run_all.sh
```

## 6. Current Core Assumptions (V1)
- Anchor regime (2-5y): NAIRU 4.2, inflation target 2.1, neutral real 0.8, term premium 1.2, mortgage spread 1.7, productivity 1.2, working-age growth 0.4, population growth 0.6.
- Mild adverse: unemployment +2.5pp, HPI growth -3pp, mortgage +1pp, GDP -1.5pp, HY spread +2pp.
- Severe adverse: unemployment +5.5pp, HPI growth -7pp, mortgage +2pp, GDP -3pp, HY spread +4pp, delinquency +1.5pp.
- Demographic low-growth: persistent negative shifts in workforce/population/GDP/HPI/retail anchor drift.

## 7. Risk Priority Legend
- **P1:** High risk issue. Can invalidate interpretation or trust.
- **P2:** Medium risk issue. Weakens quality or evidence strength.
- **P3:** Low risk issue. Hygiene/operational friction.

## 8. Main V1 Issues (What to Improve)

### P1 - Scenario timing starts one quarter late
- Expected from config: scenario starts at Q21.
- Actual in output: first non-zero scenario deviation appears at Q22.
- Code area: `scripts/run_macro_forecast_engine.py` (`triangular_shock` / scenario application loop).
- Impact: stress timing semantics are wrong by one quarter.

### P1 - Backtest spec mismatch vs production spec
- Production lag is selected by VAR criteria, then applied to BVAR.
- Backtest hard-sets lag to `max(lag_candidates)`.
- Code area:
  - Production: `scripts/run_macro_forecast_engine.py`
  - Backtest: `scripts/backtest_bvar_oos.py`
- Impact: backtest does not validate exactly what production runs.

### P2 - Backtest sample is too small for strong claims
- Current artifact has `n_oos = 8`.
- DM p-values with this sample are weak evidence.
- Impact: cannot confidently claim robust outperformance.

### P2 - Validation is not in default pipeline
- `run_all.sh` does not execute backtest/validation gate.
- Impact: outputs can be generated without passing quality checks.

### P2 - Backtest default scope is benchmark subset only
- Default variable set is benchmark variables, not full production variable list.
- Impact: evaluation coverage is narrower than production usage.

### P3 - Runtime consistency/hygiene gaps
- README uses `python3`, `run_all.sh` uses `python`.
- No `.gitignore` baseline for OS/cache junk.


## Inputs
- Config: `macro_engine_config.json`
- Data output path for panel build:
  - `data/macro_panel_quarterly_raw.csv`
  - `data/macro_panel_quarterly_model.csv`
  - `data/macro_panel_metadata.json`

## Run commands
```bash
python3 scripts/fetch_macro_panel_fred.py \
  --raw-output data/macro_panel_quarterly_raw.csv \
  --model-output data/macro_panel_quarterly_model.csv \
  --metadata-output data/macro_panel_metadata.json

python3 scripts/run_macro_forecast_engine.py \
  --config macro_engine_config.json \
  --output-dir outputs
```

## Outputs
- `outputs/macro_forecast_paths.csv`
- `outputs/macro_forecast_short_horizon_intervals.csv`
- `outputs/macro_model_diagnostics.json`
- `outputs/macro_anchor_assumptions.json`
- `outputs/macro_impulse_responses.csv`

## SchemA
<!-- SCHEMA:START -->

| Column | Type | Units | Description | Source/Transform |
|---|---|---|---|---|
| `scenario` | categorical | n/a | Forecast scenario label (Baseline, Mild_Adverse, Severe_Adverse, Demographic_LowGrowth). | scenario envelope code |
| `forecast_q` | integer | count | Quarter index in the 80-quarter horizon (1=next quarter). | generated |
| `quarter_end` | date | YYYY-MM-DD | Quarter-end date of the forecast row. | `forecast_dates` |
| `unemployment_rate` | numeric | percent | Quarterly unemployment rate (level). | FRED `UNRATE` |
| `labor_force_participation` | numeric | percent | Labor-force participation rate (level). | FRED `CIVPART` |
| `payroll_growth_yoy` | numeric | percent | YoY payroll change based on `PAYEMS`. | YoY pct |
| `wage_growth_yoy` | numeric | percent | YoY hourly wage change (`CES3000000008`). | YoY pct |
| `headline_cpi_yoy` | numeric | percent | Headline CPI YoY (`CPIAUCSL`). | YoY pct |
| `core_cpi_yoy` | numeric | percent | Core CPI YoY (`CPILFESL`). | YoY pct |
| `pce_inflation_yoy` | numeric | percent | PCE index YoY (`PCEPI`). | YoY pct |
| `oer_inflation_yoy` | numeric | percent | Owners’ equivalent rent YoY (`CUSR0000SEHC`). | YoY pct |
| `medical_cpi_yoy` | numeric | percent | Medical CPI YoY (`CPIMEDSL`). | YoY pct |
| `real_gdp_growth_yoy` | numeric | percent | Real GDP YoY (`GDPC1`). | YoY pct |
| `industrial_production_yoy` | numeric | percent | Industrial production YoY (`INDPRO`). | YoY pct |
| `consumer_sentiment` | numeric | index | Michigan Consumer Sentiment (`UMCSENT`). | level |
| `retail_sales_yoy` | numeric | percent | Retail sales YoY (`RSAFS`). | YoY pct |
| `hpi_growth_yoy` | numeric | percent | FHFA HPI YoY (`USSTHPI`). | YoY pct |
| `housing_starts_yoy` | numeric | percent | Housing starts YoY (`HOUST`). | YoY pct |
| `building_permits_yoy` | numeric | percent | Building permits YoY (`PERMIT`). | YoY pct |
| `months_supply_homes` | numeric | months | Months’ supply of homes (`MSACSR`). | level |
| `rent_inflation_yoy` | numeric | percent | Rent inflation YoY (`CUSR0000SEHA`). | YoY pct |
| `mortgage30_rate` | numeric | percent | 30-year mortgage rate (`MORTGAGE30US`). | level |
| `ust10_rate` | numeric | percent | 10-year UST yield (`GS10`). | level |
| `high_yield_spread` | numeric | percent | BAML HY spread (`BAMLH0A0HYM2`). | level |
| `prime_rate` | numeric | percent | Prime bank loan rate (`DPRIME`). | level |
| `fed_funds_rate` | numeric | percent | Fed funds effective rate (`FEDFUNDS`). | level |
| `household_credit_growth_yoy` | numeric | percent | Household credit card/auto YoY (`CMDEBT`). | YoY pct |
| `household_networth_growth_yoy` | numeric | percent | Household net worth YoY (`BOGZ1FL192090005Q`). | YoY pct |
| `consumer_delinquency_rate` | numeric | percent | Consumer delinquency rate (`DRCCLACBS`). | level |
| `working_age_pop_growth_yoy` | numeric | percent | Working-age population growth (`LFWA64TTUSM647S`). | YoY pct |
| `population_growth_yoy` | numeric | percent | Total population growth (`POPTHM`). | YoY pct |

<!-- SCHEMA:END -->


