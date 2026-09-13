import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_backend
from scipy.stats import loguniform, randint, t
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor


N = 500
P = 50
N_ACTIVE = 10
N_REPS = 25
N_JOBS = 4
MASTER_SEED = 1337
TEST_FRACTION = 0.2
TARGET_SIGNAL_SCALE = 1.0
TARGET_NOISE_SD = 1.0
TARGET_INTERCEPT = 0.0
LOWER_Q = 0.10
UPPER_Q = 0.90

OUTPUT_DIR = Path("../results")
EXCEL_PATH = OUTPUT_DIR / f"matched_nonlinearity_quantile_buckets_n{N}_p{P}_a{N_ACTIVE}.xlsx"

LEVELS_TO_RUN = [0.05, 0.10, 0.15, 0.20, 0.30]
MATCHED_STRENGTHS = {
    "cubic": {0.05: 0.1302579, 0.10: 0.22996560, 0.15: 0.3532381,
              0.20: 0.5265986, 0.30: 1.3483315},
    "quintic": {0.05: 0.0098076, 0.10: 0.0152675, 0.15: 0.0204604,
                0.20: 0.0258628, 0.30: 0.038476},
    "exp": {0.05: 0.3189425, 0.10: 0.4551335, 0.15: 0.5627497,
            0.20: 0.6563857, 0.30: 0.8218707},
    "quad": {0.05: 0.1659280, 0.10: 0.2476384, 0.15: 0.3223790,
             0.20: 0.3994894, 0.30: 0.5912237},
}

CONDITIONS = {
    "BB": ("base", "base"),
    "BT": ("base", "transformed"),
    "TB": ("transformed", "base"),
    "TT": ("transformed", "transformed"),
}
MODELS = ["xgb", "ann"]

CV = KFold(n_splits=5, shuffle=True, random_state=2025)
N_ITER_XGB = 10
N_ITER_ANN = 10
ANN_MAX_ITER = 500
PARAM_DISTRIBUTIONS_XGB = {
    "n_estimators": randint(10, 1001),
    "learning_rate": loguniform(1e-3, 1e-1),
    "max_depth": randint(1, 6),
}
PARAM_DISTRIBUTIONS_ANN = {
    "mlp__hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32)],
    "mlp__alpha": loguniform(1e-5, 1e-2),
    "mlp__learning_rate_init": loguniform(1e-4, 1e-2),
    "mlp__batch_size": [32, 64, 128],
}

GROUP_COLUMNS = ["n", "p", "n_active", "transform", "nonlinearity_level", "strength", "model"]
KEY_COLUMNS = GROUP_COLUMNS + ["rep", "condition"]


def transform(z, family, a):
    z = np.asarray(z, dtype=float)
    if family == "cubic":
        return z + a * z**3
    if family == "quintic":
        return z + a * z**5
    if family == "exp":
        return z if a == 0 else np.expm1(a * z) / a
    if family == "quad":
        return z + a * z**2 + (a**2 / 3.0) * z**3
    raise ValueError(f"Unknown transform: {family}")


def population_nonlinearity(family, a):
    if family == "cubic":
        return 6 * a**2 / (1 + 6 * a + 15 * a**2)
    if family == "quintic":
        return 720 * a**2 / (1 + 30 * a + 945 * a**2)
    if family == "exp":
        return 0.0 if a == 0 else 1 - a**2 / np.expm1(a**2)
    if family == "quad":
        return (2 * a**2 + 2 * a**4 / 3) / (1 + 4 * a**2 + 5 * a**4 / 3)
    raise ValueError(f"Unknown transform: {family}")


def standardize(a):
    sd = a.std(axis=0, ddof=0)
    if not np.isfinite(a).all() or np.any(sd <= 0):
        raise ValueError("Nonfinite data or a constant generated feature.")
    return (a - a.mean(axis=0)) / sd


def fit_model(model, X_train, X_test, y_train, y_test, seed):
    if model == "xgb":
        estimator = XGBRegressor(
            objective="reg:squarederror", tree_method="hist",
            random_state=2020, n_jobs=1, verbosity=0,
        )
        param_distributions = PARAM_DISTRIBUTIONS_XGB
        n_iter = N_ITER_XGB
    else:
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                random_state=seed, early_stopping=True,
                n_iter_no_change=15, max_iter=ANN_MAX_ITER,
            )),
        ])
        param_distributions = PARAM_DISTRIBUTIONS_ANN
        n_iter = N_ITER_ANN

    search = RandomizedSearchCV(
        estimator=estimator, param_distributions=param_distributions,
        n_iter=n_iter, scoring="neg_mean_squared_error", cv=CV,
        n_jobs=N_JOBS, pre_dispatch=N_JOBS,
        random_state=seed + 99, error_score="raise", refit=True,
    )
    with threadpool_limits(limits=1), parallel_backend("loky", inner_max_num_threads=1):
        search.fit(X_train, y_train)
        y_pred = search.best_estimator_.predict(X_test)
        y_pred_train = search.best_estimator_.predict(X_train)

    if not np.isfinite(y_pred).all() or not np.isfinite(search.best_score_):
        raise ValueError("Nonfinite predictions or cross-validation score.")
    fitted = {
        "y_pred": y_pred,
        "r2_train": r2_score(y_train, y_pred_train),
        "mse_train": mean_squared_error(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred),
        "mse_test": mean_squared_error(y_test, y_pred),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred)),
        "cv_best_mse": -search.best_score_,
        "best_params": json.dumps(search.best_params_, sort_keys=True, default=lambda x: x.item()),
    }
    fitted["generalization_gap_mse"] = fitted["mse_test"] - fitted["mse_train"]
    if model == "ann":
        mlp = search.best_estimator_.named_steps["mlp"]
        fitted.update({
            "n_iter_fitted": mlp.n_iter_,
            "hit_max_iter": mlp.n_iter_ >= ANN_MAX_ITER,
            "best_validation_score": mlp.best_validation_score_,
        })
    return fitted


def bucket_labels(values, q_low, q_high):
    return np.where(values < q_low, "low", np.where(values > q_high, "high", "middle"))


def bucket_metrics(y, pred, signal, labels):
    rows = []
    for bucket in ("low", "middle", "high", "overall"):
        mask = np.ones(len(y), dtype=bool) if bucket == "overall" else labels == bucket
        n = int(mask.sum())
        row = {"bucket": bucket, "n_bucket": n}
        for prefix, error in (("", pred - y), ("signal_", pred - signal), ("oracle_", signal - y)):
            e = error[mask]
            sse = float(np.dot(e, e))
            absolute_sum = float(np.abs(e).sum())
            error_sum = float(e.sum())
            row.update({
                prefix + "sse": sse,
                prefix + "absolute_error_sum": absolute_sum,
                prefix + "error_sum": error_sum,
                prefix + "mse": sse / n if n else None,
                prefix + "rmse": float(np.sqrt(sse / n)) if n else None,
                prefix + "mae": absolute_sum / n if n else None,
                prefix + "bias": error_sum / n if n else None,
            })
        row["median_absolute_error"] = float(np.median(np.abs((pred - y)[mask]))) if n else None
        rows.append(row)
    return rows


def paired_results(buckets):
    if buckets.empty:
        return pd.DataFrame()
    keys = GROUP_COLUMNS + ["rep", "bucket_by", "bucket"]
    parts = []
    for mismatch, reference in (("TB", "BB"), ("BT", "TT")):
        joined = buckets[buckets.condition == mismatch].merge(
            buckets[buckets.condition == reference], on=keys,
            suffixes=("_mismatch", "_reference"), validate="one_to_one",
        )
        if joined.empty:
            continue
        for field in ("n_bucket", "q_low", "q_high", "test_sst", "oracle_sse"):
            if not np.allclose(joined[field + "_mismatch"], joined[field + "_reference"], equal_nan=True):
                raise ValueError(f"Pairing mismatch in {field} for {mismatch}-{reference}.")
        out = joined[keys].copy()
        out["comparison"] = mismatch + "-" + reference
        out["n_bucket"] = joined.n_bucket_mismatch
        for metric in ("mse", "rmse", "bias", "signal_mse", "signal_rmse", "signal_bias"):
            out["delta_" + metric] = joined[metric + "_mismatch"] - joined[metric + "_reference"]
        out["delta_sse"] = joined.sse_mismatch - joined.sse_reference
        out["r2_loss_contribution"] = out.delta_sse / joined.test_sst_reference
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def pooled_summary(buckets):
    if buckets.empty:
        return pd.DataFrame()
    keys = GROUP_COLUMNS + ["condition", "bucket_by", "bucket"]
    components = ["n_bucket"] + [
        prefix + name for prefix in ("", "signal_", "oracle_")
        for name in ("sse", "absolute_error_sum", "error_sum")
    ]
    grouped = buckets.groupby(keys, dropna=False)
    out = grouped[components].sum()
    out["n_reps"] = grouped.rep.nunique()
    out["n_empty_reps"] = grouped.n_bucket.apply(lambda x: int((x == 0).sum()))
    count = out.n_bucket.replace(0, np.nan)
    for prefix in ("", "signal_", "oracle_"):
        out[prefix + "mse_pooled"] = out[prefix + "sse"] / count
        out[prefix + "rmse_pooled"] = np.sqrt(out[prefix + "mse_pooled"])
        out[prefix + "mae_pooled"] = out[prefix + "absolute_error_sum"] / count
        out[prefix + "bias_pooled"] = out[prefix + "error_sum"] / count
    return out.reset_index()


def paired_summary(pairs):
    if pairs.empty:
        return pd.DataFrame()
    keys = GROUP_COLUMNS + ["comparison", "bucket_by", "bucket"]
    rows = []
    metrics = [name for name in pairs if name.startswith("delta_")] + ["r2_loss_contribution"]
    for values, group in pairs.groupby(keys, dropna=False):
        row = dict(zip(keys, values))
        row["n_paired_reps"] = int(group.rep.nunique())
        row["n_nonempty_pairs"] = int((group.n_bucket > 0).sum())
        row["n_test_points"] = int(group.n_bucket.sum())
        for metric in metrics:
            data = group[metric].dropna().to_numpy(dtype=float)
            n = len(data)
            mean = float(data.mean()) if n else np.nan
            se = float(data.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            margin = float(t.ppf(0.975, n - 1) * se) if n > 1 else np.nan
            row.update({metric + "_mean": mean, metric + "_se": se,
                        metric + "_ci95_low": mean - margin, metric + "_ci95_high": mean + margin})
        rows.append(row)
    return pd.DataFrame(rows)


def key_tuple(row):
    return tuple(row[column] for column in KEY_COLUMNS)


def load_existing_results():
    if not EXCEL_PATH.exists():
        return {}, {}
    results = pd.read_excel(EXCEL_PATH, sheet_name="results").to_dict("records")
    buckets = pd.read_excel(EXCEL_PATH, sheet_name="buckets").to_dict("records")
    results = {key_tuple(row): row for row in results}
    buckets = {key_tuple(row) + (row["bucket_by"], row["bucket"]): row for row in buckets}
    return results, buckets


def save_progress(results, buckets):
    result_frame = pd.DataFrame(results.values()).sort_values(KEY_COLUMNS)
    bucket_frame = pd.DataFrame(buckets.values())
    if not bucket_frame.empty:
        bucket_frame = bucket_frame.sort_values(KEY_COLUMNS + ["bucket_by", "bucket"])
    pairs = paired_results(bucket_frame)
    temporary_path = EXCEL_PATH.with_name(EXCEL_PATH.stem + ".tmp.xlsx")
    with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
        result_frame.to_excel(writer, sheet_name="results", index=False)
        bucket_frame.to_excel(writer, sheet_name="buckets", index=False)
        pooled_summary(bucket_frame).to_excel(writer, sheet_name="bucket_summary", index=False)
        pairs.to_excel(writer, sheet_name="paired", index=False)
        paired_summary(pairs).to_excel(writer, sheet_name="paired_summary", index=False)
    temporary_path.replace(EXCEL_PATH)
    print(f"Saved {len(results):,} results to {EXCEL_PATH}", flush=True)


def run_experiment():
    EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    results, buckets = load_existing_results()
    completed = {key for key, row in results.items() if row["status"] == "ok"}
    weights = np.r_[np.ones(N_ACTIVE), np.zeros(P - N_ACTIVE)]

    for rep in range(1, N_REPS + 1):
        seed = MASTER_SEED + rep
        Z = np.random.default_rng(seed).standard_normal((N, P))
        base = standardize(Z)
        split_rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, rep, 101]))
        noise_rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, rep, 7]))
        permutation = split_rng.permutation(N)
        test_size = int(np.ceil(TEST_FRACTION * N))
        test_idx, train_idx = permutation[:test_size], permutation[test_size:]
        noise = noise_rng.normal(0, TARGET_NOISE_SD, N)
        bb_fits = {}

        for transform_name, strengths in MATCHED_STRENGTHS.items():
            for level in LEVELS_TO_RUN:
                strength = strengths[level]
                transformed = standardize(transform(Z, transform_name, strength))
                representations = {"base": base, "transformed": transformed}
                sample_l = np.maximum(0.0, 1 - np.mean(base * transformed, axis=0)**2)[:N_ACTIVE]

                for condition, (observed_name, signal_name) in CONDITIONS.items():
                    X = representations[observed_name]
                    signal = TARGET_INTERCEPT + TARGET_SIGNAL_SCALE * (representations[signal_name] @ weights)
                    y = signal + noise
                    y_test, signal_test = y[test_idx], signal[test_idx]
                    test_sst = np.sum((y_test - y_test.mean())**2)

                    for model in MODELS:
                        meta = {
                            "n": N, "p": P, "n_active": N_ACTIVE,
                            "transform": transform_name, "nonlinearity_level": level,
                            "strength": strength, "population_nonlinearity": population_nonlinearity(transform_name, strength),
                            "rep": rep, "seed": seed, "condition": condition, "model": model,
                            "observed_representation": observed_name, "signal_representation": signal_name,
                            "sample_nonlinearity_active_mean": sample_l.mean(),
                            "sample_nonlinearity_active_min": sample_l.min(),
                            "sample_nonlinearity_active_max": sample_l.max(),
                        }
                        key = key_tuple(meta)
                        if key in completed:
                            continue
                        print(f"rep={rep}/{N_REPS} {transform_name} L={level:.2f} {condition} {model}", flush=True)

                        try:
                            reused = condition == "BB" and model in bb_fits
                            if reused:
                                fitted = bb_fits[model]
                            else:
                                fitted = fit_model(model, X[train_idx], X[test_idx], y[train_idx], y_test, seed)
                            y_pred = fitted["y_pred"]
                            bucket_rows = []
                            for bucket_by, values in (("signal", signal), ("observed_y", y)):
                                q_low, q_high = np.quantile(values[train_idx], [LOWER_Q, UPPER_Q])
                                labels = bucket_labels(values[test_idx], q_low, q_high)
                                for row in bucket_metrics(y_test, y_pred, signal_test, labels):
                                    bucket_rows.append({**meta, "bucket_by": bucket_by,
                                                        "q_low": q_low, "q_high": q_high,
                                                        "test_sst": test_sst, **row})
                            results[key] = {
                                **meta, **{k: v for k, v in fitted.items() if k != "y_pred"},
                                "status": "ok", "error_message": "", "bb_fit_reused": reused,
                                "n_train": len(train_idx), "n_test": len(test_idx), "test_sst": test_sst,
                                "signal_variance_full": signal.var(),
                                "signal_variance_train": signal[train_idx].var(),
                                "signal_variance_test": signal_test.var(),
                                "mse_to_signal_test": np.mean((y_pred - signal_test)**2),
                                "oracle_mse_test": np.mean((signal_test - y_test)**2),
                            }
                            for row in bucket_rows:
                                buckets[key + (row["bucket_by"], row["bucket"])] = row
                            completed.add(key)
                            if condition == "BB":
                                bb_fits[model] = fitted
                        except Exception as exc:
                            results[key] = {**meta, "status": "failed", "error_message": str(exc)}
                            print(f"Failed: {exc}", flush=True)

        save_progress(results, buckets)

    failed = sum(row["status"] == "failed" for row in results.values())
    print(f"Finished. Failed fits: {failed}.", flush=True)


if __name__ == "__main__":
    run_experiment()
