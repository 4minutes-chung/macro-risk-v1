#!/usr/bin/env python3
"""
Rolling out-of-sample backtest for the macro BVAR.

Outputs:
- bvar_oos_backtest_table.csv: RMSE, MAE, interval coverage, DM statistic/p-value
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class VarModel:
    intercept: np.ndarray
    coefs: np.ndarray
    sigma: np.ndarray
    lag_order: int
    variables: List[str]


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest BVAR short-horizon forecast power.")
    parser.add_argument("--config", default="macro_engine_config.json", help="Config JSON path.")
    parser.add_argument(
        "--input",
        default="data/macro_panel_quarterly_model.csv",
        help="Model panel CSV path.",
    )
    parser.add_argument(
        "--output",
        default="outputs/macro_engine/bvar_oos_backtest_table.csv",
        help="Output table CSV path.",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=80,
        help="Minimum training quarters before first origin.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=8,
        help="Forecast horizon in quarters for backtest.",
    )
    parser.add_argument(
        "--interval-sims",
        type=int,
        default=300,
        help="Simulation paths for predictive interval coverage.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--variable-set",
        choices=["benchmark", "full"],
        default="benchmark",
        help="Use benchmark variable subset (fast) or full variable list (slow).",
    )
    parser.add_argument(
        "--max-origins",
        type=int,
        default=40,
        help="Maximum number of rolling origins to evaluate (newest origins kept).",
    )
    parser.add_argument(
        "--origin-stride",
        type=int,
        default=1,
        help="Use every k-th rolling origin to speed up runtime.",
    )
    return parser.parse_args()


def make_lagged_xy(y: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    t, n = y.shape
    rows = t - p
    x = np.ones((rows, 1 + n * p), dtype=float)
    y_target = y[p:, :]
    for lag in range(1, p + 1):
        x[:, 1 + (lag - 1) * n: 1 + lag * n] = y[p - lag: t - lag, :]
    return x, y_target


def fit_bvar_minnesota(
    y: np.ndarray,
    p: int,
    variables: List[str],
    level_vars: List[bool],
    lambda1: float,
    lambda2: float,
    lambda3: float,
    lambda4: float,
) -> Tuple[VarModel, List[np.ndarray]]:
    x, y_target = make_lagged_xy(y, p)
    t_eff, k = x.shape
    n = y.shape[1]

    sigma_scale = np.var(y_target, axis=0, ddof=1)
    sigma_scale = np.where(sigma_scale <= 1e-8, 1.0, sigma_scale)

    beta_mean = np.zeros((k, n), dtype=float)
    beta_cov_blocks = []
    sigma_resid = np.zeros((n, n), dtype=float)
    xtx = x.T @ x

    for j in range(n):
        b0 = np.zeros(k, dtype=float)
        if level_vars[j]:
            b0[1 + j] = 1.0

        prior_var = np.zeros(k, dtype=float)
        prior_var[0] = (lambda4 ** 2) * sigma_scale[j]
        for lag in range(1, p + 1):
            for m in range(n):
                idx = 1 + (lag - 1) * n + m
                tight = (lambda1 ** 2) / (lag ** (2.0 * lambda3))
                if m != j:
                    tight *= (lambda2 ** 2)
                tight *= sigma_scale[j] / sigma_scale[m]
                prior_var[idx] = tight
        prior_var = np.where(prior_var <= 1e-10, 1e-10, prior_var)

        prior_prec = np.diag(1.0 / prior_var)
        post_prec = xtx + prior_prec
        post_cov_core = np.linalg.inv(post_prec)
        rhs = x.T @ y_target[:, j] + prior_prec @ b0
        beta_j = post_cov_core @ rhs

        resid_j = y_target[:, j] - x @ beta_j
        sig_j = float((resid_j @ resid_j) / max(1, t_eff - k))
        sigma_resid[j, j] = max(sig_j, 1e-8)
        beta_mean[:, j] = beta_j
        beta_cov_blocks.append(post_cov_core * sigma_resid[j, j])

    intercept = beta_mean[0, :]
    coefs = np.zeros((p, n, n), dtype=float)
    for lag in range(1, p + 1):
        block = beta_mean[1 + (lag - 1) * n: 1 + lag * n, :]
        coefs[lag - 1, :, :] = block.T

    return VarModel(
        intercept=intercept,
        coefs=coefs,
        sigma=np.diag(np.diag(sigma_resid)),
        lag_order=p,
        variables=variables,
    ), beta_cov_blocks


def deterministic_forecast(model: VarModel, history: np.ndarray, horizon: int) -> np.ndarray:
    p = model.lag_order
    hist = history.copy()
    n = history.shape[1]
    out = np.zeros((horizon, n), dtype=float)
    for h in range(horizon):
        y_next = model.intercept.copy()
        for lag in range(1, p + 1):
            y_next += model.coefs[lag - 1, :, :] @ hist[-lag, :]
        out[h, :] = y_next
        hist = np.vstack([hist, y_next])
    return out


def simulate_intervals(
    model: VarModel,
    history: np.ndarray,
    horizon: int,
    n_sims: int,
    seed: int,
    beta_cov_blocks: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = model.lag_order
    n = history.shape[1]
    k = 1 + n * p

    beta_mean_matrix = np.zeros((k, n), dtype=float)
    beta_mean_matrix[0, :] = model.intercept
    for lag in range(1, p + 1):
        beta_mean_matrix[1 + (lag - 1) * n: 1 + lag * n, :] = model.coefs[lag - 1, :, :].T

    sims = np.zeros((n_sims, horizon, n), dtype=float)
    for s in range(n_sims):
        sampled_beta = np.zeros_like(beta_mean_matrix)
        for j in range(n):
            sampled_beta[:, j] = rng.multivariate_normal(beta_mean_matrix[:, j], beta_cov_blocks[j])

        sampled_intercept = sampled_beta[0, :]
        sampled_coefs = np.zeros((p, n, n), dtype=float)
        for lag in range(1, p + 1):
            block = sampled_beta[1 + (lag - 1) * n: 1 + lag * n, :]
            sampled_coefs[lag - 1, :, :] = block.T

        hist = history.copy()
        for h in range(horizon):
            eps = rng.multivariate_normal(np.zeros(n), model.sigma)
            y_next = sampled_intercept.copy()
            for lag in range(1, p + 1):
                y_next += sampled_coefs[lag - 1, :, :] @ hist[-lag, :]
            y_next += eps
            sims[s, h, :] = y_next
            hist = np.vstack([hist, y_next])

    q05 = np.quantile(sims, 0.05, axis=0)
    q95 = np.quantile(sims, 0.95, axis=0)
    return q05, q95


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def dm_test(loss_model: np.ndarray, loss_benchmark: np.ndarray) -> Tuple[float, float]:
    d = loss_model - loss_benchmark
    t = d.shape[0]
    if t < 5:
        return float("nan"), float("nan")
    d_bar = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    if var_d <= 1e-12:
        return float("nan"), float("nan")
    stat = d_bar / math.sqrt(var_d / t)
    pvalue = 2.0 * (1.0 - norm_cdf(abs(stat)))
    return stat, pvalue


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    panel = pd.read_csv(args.input, parse_dates=["quarter_end"]).sort_values("quarter_end")
    full_variables = config["model"]["variables"]
    benchmark_variables = config["model"].get("var_benchmark_variables", full_variables)
    variables = benchmark_variables if args.variable_set == "benchmark" else full_variables
    panel = panel[["quarter_end"] + variables].dropna().copy()
    y_all = panel[variables].to_numpy(dtype=float)
    dates = panel["quarter_end"].to_numpy()

    lag_candidates = [int(x) for x in config["model"]["lag_candidates"]]
    p = max(lag_candidates)
    level_flags = [bool(config["model"]["level_variable_flags"].get(v, False)) for v in variables]
    bvar_cfg = config["model"]["bvar_hyperparams"]

    h_max = int(args.horizon)
    min_train = max(int(args.min_train), p + 12)
    if y_all.shape[0] < min_train + h_max + 5:
        raise RuntimeError("Not enough observations for requested min-train and horizon.")

    n = len(variables)
    errs_bvar = {h: [] for h in range(1, h_max + 1)}
    errs_rw = {h: [] for h in range(1, h_max + 1)}
    cover = {h: [] for h in range(1, h_max + 1)}

    start_origin = min_train
    end_origin = y_all.shape[0] - h_max
    origin_candidates = list(range(start_origin, end_origin, max(1, int(args.origin_stride))))
    if int(args.max_origins) > 0 and len(origin_candidates) > int(args.max_origins):
        origin_candidates = origin_candidates[-int(args.max_origins):]

    for idx_origin, origin in enumerate(origin_candidates, start=1):
        if idx_origin % 5 == 0:
            print(f"Backtest progress: {idx_origin}/{len(origin_candidates)} origins", flush=True)
        y_train = y_all[:origin, :]
        model, beta_cov = fit_bvar_minnesota(
            y=y_train,
            p=p,
            variables=variables,
            level_vars=level_flags,
            lambda1=float(bvar_cfg["lambda1"]),
            lambda2=float(bvar_cfg["lambda2"]),
            lambda3=float(bvar_cfg["lambda3"]),
            lambda4=float(bvar_cfg["lambda4"]),
        )

        hist = y_train[-p:, :]
        fcst = deterministic_forecast(model, hist, h_max)
        if int(args.interval_sims) > 0:
            q05, q95 = simulate_intervals(
                model=model,
                history=hist,
                horizon=h_max,
                n_sims=int(args.interval_sims),
                seed=int(args.seed) + origin,
                beta_cov_blocks=beta_cov,
            )
        else:
            q05, q95 = None, None

        last_obs = y_train[-1, :]
        for h in range(1, h_max + 1):
            actual = y_all[origin + h - 1, :]
            pred = fcst[h - 1, :]
            err_b = actual - pred
            err_rw = actual - last_obs
            errs_bvar[h].append(err_b)
            errs_rw[h].append(err_rw)
            if q05 is not None and q95 is not None:
                inside = (actual >= q05[h - 1, :]) & (actual <= q95[h - 1, :])
                cover[h].append(inside.astype(float))

    rows = []
    for h in range(1, h_max + 1):
        e_b = np.vstack(errs_bvar[h])
        e_rw = np.vstack(errs_rw[h])
        c_h = np.vstack(cover[h]) if cover[h] else None
        for j, var in enumerate(variables):
            loss_b = e_b[:, j] ** 2
            loss_rw = e_rw[:, j] ** 2
            dm_stat, dm_p = dm_test(loss_b, loss_rw)
            rows.append(
                {
                    "variable": var,
                    "horizon_q": h,
                    "n_oos": int(e_b.shape[0]),
                    "rmse_bvar": float(np.sqrt(np.mean(loss_b))),
                    "mae_bvar": float(np.mean(np.abs(e_b[:, j]))),
                    "coverage_90_bvar": float(np.mean(c_h[:, j])) if c_h is not None else np.nan,
                    "dm_stat_bvar_minus_rw_mse": float(dm_stat) if not np.isnan(dm_stat) else np.nan,
                    "dm_pvalue_two_sided": float(dm_p) if not np.isnan(dm_p) else np.nan,
                    "rmse_rw": float(np.sqrt(np.mean(loss_rw))),
                }
            )

    out = pd.DataFrame(rows).sort_values(["variable", "horizon_q"])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Wrote backtest table: {args.output}")
    print(f"OOS origins: {len(origin_candidates)}; variables: {n}; horizon: {h_max}")
    print(f"Sample date span: {pd.to_datetime(dates[0]).date()} to {pd.to_datetime(dates[-1]).date()}")


if __name__ == "__main__":
    main()
