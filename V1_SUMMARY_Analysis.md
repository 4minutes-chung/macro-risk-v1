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
