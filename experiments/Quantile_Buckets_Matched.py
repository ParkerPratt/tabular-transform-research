import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import time
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
ROOT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = ROOT_DIR / "../results"

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

SEARCH_SPECS = {
    "xgb": {
        "n_estimators": {"randint": [10, 1001]},
        "learning_rate": {"loguniform": [1e-3, 1e-1]},
        "max_depth": {"randint": [1, 6]},
    },
    "ann": {
        "mlp__hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32)],
        "mlp__alpha": {"loguniform": [1e-5, 1e-2]},
        "mlp__learning_rate_init": {"loguniform": [1e-4, 1e-2]},
        "mlp__batch_size": [32, 64, 128],
    },
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


def make_replication(config, rep):
    seed = config["master_seed"] + rep
    z = np.random.default_rng(seed).standard_normal((config["n"], config["p"]))
    split_seed = np.random.SeedSequence([config["master_seed"], rep, config["split_stream_tag"]])
    noise_seed = np.random.SeedSequence([config["master_seed"], rep, config["noise_stream_tag"]])
    order = np.random.default_rng(split_seed).permutation(config["n"])
    n_test = int(np.ceil(config["test_fraction"] * config["n"]))
    noise = np.random.default_rng(noise_seed).normal(0, config["noise_sd"], config["n"])
    weights = np.r_[np.ones(config["n_active"]), np.zeros(config["p"] - config["n_active"])]
    return z, standardize(z), noise, weights, order[n_test:], order[:n_test], seed


def search_space(spec):
    space = {}
    for name, value in spec.items():
        if isinstance(value, dict):
            kind, bounds = next(iter(value.items()))
            space[name] = {"randint": randint, "loguniform": loguniform}[kind](*bounds)
        elif name.endswith("hidden_layer_sizes"):
            space[name] = [tuple(shape) for shape in value]
        else:
            space[name] = value
    return space


def fit_model(model, x_train, x_test, y_train, y_test, seed, config, jobs):
    if model == "xgb":
        from xgboost import XGBRegressor
        estimator = XGBRegressor(**config["xgb_params"])
    else:
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(random_state=seed, **config["ann_params"])),
        ])
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=search_space(config["search_specs"][model]),
        n_iter=config["search_iterations"],
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=config["cv_folds"], shuffle=True, random_state=2025),
        n_jobs=jobs,
        pre_dispatch=jobs,
        random_state=seed + 99,
        error_score="raise",
        refit=True,
    )
    started = time.perf_counter()
    with threadpool_limits(limits=1), parallel_backend("loky", inner_max_num_threads=1):
        search.fit(x_train, y_train)
        fitted = search.best_estimator_
        pred_train = fitted.predict(x_train)
        pred = fitted.predict(x_test)
    if not np.isfinite(pred).all() or not np.isfinite(pred_train).all():
        raise ValueError("Nonfinite predictions.")
    if not np.isfinite(search.best_score_):
        raise ValueError("Nonfinite cross-validation score.")
    metrics = {
        "r2_train": float(r2_score(y_train, pred_train)),
        "mse_train": float(mean_squared_error(y_train, pred_train)),
        "r2_test": float(r2_score(y_test, pred)),
        "mse_test": float(mean_squared_error(y_test, pred)),
        "cv_best_mse": float(-search.best_score_),
        "best_params": json.dumps(clean_json(search.best_params_), sort_keys=True),
        "fit_seconds": time.perf_counter() - started,
    }
    metrics["rmse_test"] = float(np.sqrt(metrics["mse_test"]))
    metrics["generalization_gap_mse"] = metrics["mse_test"] - metrics["mse_train"]
    if model == "ann":
        mlp = fitted.named_steps["mlp"]
        metrics.update({
            "n_iter_fitted": int(mlp.n_iter_),
            "hit_max_iter": bool(mlp.n_iter_ >= config["ann_params"]["max_iter"]),
            "best_validation_score": float(mlp.best_validation_score_),
        })
    return {"metrics": metrics, "predictions": pred.tolist()}


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


def make_record(meta, fit, y, signal, train, test, reused):
    pred = np.asarray(fit["predictions"])
    y_test, signal_test = y[test], signal[test]
    sst = float(np.sum((y_test - y_test.mean())**2))
    result = {
        **meta, **fit["metrics"], "status": "ok", "error_message": "",
        "bb_fit_reused": reused, "fit_seconds": 0.0 if reused else fit["metrics"]["fit_seconds"],
        "n_train": len(train), "n_test": len(test), "test_sst": sst,
        "signal_variance_full": float(signal.var()),
        "signal_variance_train": float(signal[train].var()),
        "signal_variance_test": float(signal_test.var()),
        "mse_to_signal_test": float(np.mean((pred - signal_test)**2)),
        "oracle_mse_test": float(np.mean((signal_test - y_test)**2)),
    }
    buckets, labels_by = [], {}
    for bucket_by, values in (("signal", signal), ("observed_y", y)):
        q_low, q_high = np.quantile(values[train], [LOWER_Q, UPPER_Q])
        labels = bucket_labels(values[test], q_low, q_high)
        labels_by[bucket_by] = labels.tolist()
        for row in bucket_metrics(y_test, pred, signal_test, labels):
            buckets.append({
                **meta, "bucket_by": bucket_by, "q_low": float(q_low), "q_high": float(q_high),
                "test_sst": sst, **row,
            })
    return {
        "result": result, "fit": fit, "buckets": buckets,
        "test_index": test.tolist(), "y_true": y_test.tolist(), "signal": signal_test.tolist(),
        "labels": labels_by,
    }


def clean_json(value):
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_json(value):
    return json.dumps(clean_json(value), sort_keys=True, allow_nan=False, separators=(",", ":"))


def record_key(meta):
    return canonical_json([meta[k] for k in KEY_COLUMNS])


def open_checkpoint(path, config):
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS runs (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    stored = connection.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
    serialized = canonical_json(config)
    if stored and stored[0] != serialized:
        connection.close()
        raise ValueError("Checkpoint configuration or package versions differ. Use a different --output-dir.")
    if not stored:
        with connection:
            connection.execute("INSERT INTO metadata VALUES ('config', ?)", (serialized,))
    records = {key: json.loads(payload) for key, payload in connection.execute("SELECT key, payload FROM runs")}
    return connection, records


def save_record(connection, records, record):
    key = record_key(record["result"])
    cleaned = clean_json(record)
    with connection:
        connection.execute("INSERT OR REPLACE INTO runs VALUES (?, ?)", (key, canonical_json(cleaned)))
    records[key] = cleaned


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


def export_results(records, config, workbook):
    results = pd.DataFrame([record["result"] for record in records.values()])
    buckets = pd.DataFrame([row for record in records.values() for row in record.get("buckets", [])])
    if not results.empty:
        results = results.sort_values(KEY_COLUMNS).reset_index(drop=True)
    if not buckets.empty:
        buckets = buckets.sort_values(KEY_COLUMNS + ["bucket_by", "bucket"]).reset_index(drop=True)
    pairs = paired_results(buckets)
    metadata = pd.DataFrame([{"setting": key, "value": canonical_json(value)} for key, value in config.items()])
    temporary = workbook.with_name(workbook.stem + ".tmp.xlsx")
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        for name, frame in (("results", results), ("buckets", buckets),
                            ("bucket_summary", pooled_summary(buckets)), ("paired", pairs),
                            ("paired_summary", paired_summary(pairs)), ("metadata", metadata)):
            frame.to_excel(writer, sheet_name=name, index=False)
    os.replace(temporary, workbook)
    print(f"Saved {len(results):,} result rows: {workbook}", flush=True)


def export_predictions(records, path):
    fields = KEY_COLUMNS + ["test_index", "y_true", "signal", "y_pred", "signal_bucket", "observed_y_bucket"]
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records.values():
            if record["result"]["status"] != "ok":
                continue
            key = {name: record["result"][name] for name in KEY_COLUMNS}
            columns = zip(record["test_index"], record["y_true"], record["signal"],
                          record["fit"]["predictions"], record["labels"]["signal"], record["labels"]["observed_y"])
            for values in columns:
                writer.writerow({**key, **dict(zip(fields[len(KEY_COLUMNS):], values))})
    os.replace(temporary, path)


def package_versions():
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "xgboost", "joblib", "threadpoolctl", "openpyxl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name == "xgboost":
                try:
                    versions[name] = importlib.metadata.version("xgboost-cpu")
                    continue
                except importlib.metadata.PackageNotFoundError:
                    pass
            versions[name] = "unavailable"
    return versions


def build_config(args):
    specs = clean_json(SEARCH_SPECS)
    if args.smoke_test:
        specs["xgb"]["n_estimators"] = {"randint": [10, 31]}
        specs["ann"]["mlp__batch_size"] = [16, 32]
    return {
        "schema_version": 2, "design": "matched_nonlinearity_quantile_buckets",
        "n": args.n, "p": args.p, "n_active": args.active,
        "master_seed": MASTER_SEED, "test_fraction": TEST_FRACTION,
        "signal_scale": TARGET_SIGNAL_SCALE, "noise_sd": TARGET_NOISE_SD,
        "intercept": TARGET_INTERCEPT, "latent_correlation": "identity",
        "representation_normalization": "full_draw_sample_mean_sd_ddof0",
        "transformed_features": "all", "clipping": False,
        "grid": [{"transform": family, "nonlinearity_level": level,
                  "strength": MATCHED_STRENGTHS[family][level],
                  "population_nonlinearity": float(population_nonlinearity(family, MATCHED_STRENGTHS[family][level]))}
                 for family in args.families for level in args.levels],
        "conditions": CONDITIONS, "models": args.models, "bucket_quantiles": [LOWER_Q, UPPER_Q],
        "bucket_cutoff_source": "training_only", "bucket_by": ["signal", "observed_y"],
        "search_iterations": args.search_iterations, "cv_folds": args.cv_folds,
        "search_specs": specs, "cv_seed": 2025, "search_seed_offset": 99,
        "rng_scheme": "data: master_seed+rep; noise/split: SeedSequence([master_seed,rep,stream_tag])",
        "split_stream_tag": 101, "noise_stream_tag": 7,
        "ann_params": {"early_stopping": True, "n_iter_no_change": 15, "max_iter": args.max_iter,
                       "activation": "relu", "solver": "adam", "validation_fraction": 0.1},
        "xgb_params": {"objective": "reg:squarederror", "tree_method": "hist",
                       "random_state": 2020, "n_jobs": 1, "verbosity": 0},
        "smoke_test": args.smoke_test, "versions": package_versions(),
        "paired_ci": "pointwise approximate Student t interval over independent replication deltas",
        "bias_sign": "prediction minus truth; signal_bias uses noiseless signal as truth",
        "source_design": "Matched_Nonlinearity(3).py",
    }


def validate_config(args, config):
    if args.n < 40 or not 1 <= args.active <= args.p or args.reps < 1 or args.jobs < 1:
        raise ValueError("Require n >= 40, 1 <= active <= p, reps >= 1, and jobs >= 1.")
    if args.search_iterations < 1 or args.max_iter < 1 or not 2 <= args.cv_folds <= 10:
        raise ValueError("Invalid search iterations, max_iter, or cv_folds (2 through 10).")
    if not 0 < TEST_FRACTION < 1 or not 0 < LOWER_Q < UPPER_Q < 1:
        raise ValueError("Invalid split fraction or quantile levels.")
    if TARGET_NOISE_SD < 0 or TARGET_SIGNAL_SCALE <= 0:
        raise ValueError("Noise SD must be nonnegative; signal scale must be positive.")
    n_train = args.n - int(np.ceil(TEST_FRACTION * args.n))
    smallest_cv_train = n_train - int(np.ceil(n_train / args.cv_folds))
    if int(np.ceil(0.1 * smallest_cv_train)) < 2:
        raise ValueError("Too few observations for ANN early-stopping validation.")
    for row in config["grid"]:
        if abs(row["population_nonlinearity"] - row["nonlinearity_level"]) > 1e-4:
            raise ValueError(f"Matched strength fails calibration check: {row}")


def run_experiment(args, config):
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = ("smoke_" if args.smoke_test else "") + f"matched_nonlinearity_quantile_buckets_n{args.n}_p{args.p}_a{args.active}"
    workbook = output_dir / (stem + ".xlsx")
    checkpoint = output_dir / (stem + ".sqlite")
    if args.export_only and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if workbook.exists() and not checkpoint.exists():
        raise ValueError("Workbook exists without its checkpoint. Use a different --output-dir.")
    connection, records = open_checkpoint(checkpoint, config)
    if records and max(row["result"]["rep"] for row in records.values()) > args.reps:
        connection.close()
        raise ValueError("Checkpoint contains more repetitions than --reps; increase --reps.")
    bb_cache = {(row["result"]["rep"], row["result"]["model"]): row["fit"]
                for row in records.values() if row["result"]["status"] == "ok" and row["result"]["condition"] == "BB"}
    expected = args.reps * len(config["grid"]) * 4 * len(args.models)
    unique_fits = args.reps * (1 + 3 * len(config["grid"])) * len(args.models)
    print(f"Output: {output_dir}\nPlanned: {unique_fits:,} unique searches; {expected:,} result rows.", flush=True)
    dirty = False
    try:
        if not args.export_only:
            for rep in range(1, args.reps + 1):
                z, base, noise, weights, train, test, seed = make_replication(config, rep)
                for grid_row in config["grid"]:
                    family, strength = grid_row["transform"], grid_row["strength"]
                    transformed = standardize(transform(z, family, strength))
                    representations = {"base": base, "transformed": transformed}
                    signals = {name: config["intercept"] + config["signal_scale"] * (matrix @ weights)
                               for name, matrix in representations.items()}
                    sample_l = np.maximum(0.0, 1 - np.mean(base * transformed, axis=0)**2)[:args.active]
                    for condition, (observed_name, signal_name) in CONDITIONS.items():
                        x, signal = representations[observed_name], signals[signal_name]
                        y = signal + noise
                        for model in args.models:
                            meta = {
                                "n": args.n, "p": args.p, "n_active": args.active,
                                **grid_row, "rep": rep, "seed": seed, "condition": condition,
                                "observed_representation": observed_name, "signal_representation": signal_name,
                                "model": model, "sample_nonlinearity_active_mean": float(sample_l.mean()),
                                "sample_nonlinearity_active_min": float(sample_l.min()),
                                "sample_nonlinearity_active_max": float(sample_l.max()),
                            }
                            previous = records.get(record_key(meta))
                            if previous and previous["result"]["status"] == "ok":
                                continue
                            reused = condition == "BB" and (rep, model) in bb_cache
                            print(f"rep={rep}/{args.reps} {family} L={grid_row['nonlinearity_level']:.2f} {condition} {model}"
                                  + (" [BB reused]" if reused else ""), flush=True)
                            try:
                                fit = bb_cache[(rep, model)] if reused else fit_model(
                                    model, x[train], x[test], y[train], y[test], seed, config, args.jobs)
                                record = make_record(meta, fit, y, signal, train, test, reused)
                                if condition == "BB":
                                    bb_cache[(rep, model)] = fit
                            except Exception as exc:
                                record = {"result": {**meta, "status": "failed",
                                                      "error_message": f"{type(exc).__name__}: {exc}"}}
                                print(f"FAILED: {record['result']['error_message']}", flush=True)
                            save_record(connection, records, record)
                            dirty = True
                if dirty and rep < args.reps:
                    export_results(records, config, workbook)
                    dirty = False
        export_results(records, config, workbook)
        export_predictions(records, output_dir / (stem + "_predictions.csv.gz"))
        successful = sum(row["result"]["status"] == "ok" for row in records.values())
        failed = sum(row["result"]["status"] == "failed" for row in records.values())
        print(f"Successful: {successful}/{expected}; failed: {failed}; missing: {expected - len(records)}.", flush=True)
        if not args.export_only and successful != expected:
            raise RuntimeError("Some fits failed or are missing. Rerun the same command to retry them.")
    finally:
        connection.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Quantile diagnostics for the matched-nonlinearity experiment.")
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--p", type=int, default=P)
    parser.add_argument("--active", type=int, default=N_ACTIVE)
    parser.add_argument("--reps", type=int, default=N_REPS)
    parser.add_argument("--jobs", type=int, default=N_JOBS)
    parser.add_argument("--families", nargs="+", choices=list(MATCHED_STRENGTHS), default=list(MATCHED_STRENGTHS))
    parser.add_argument("--levels", nargs="+", type=float, choices=[0.05, 0.10, 0.15, 0.20, 0.30],
                        default=[0.05, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--models", nargs="+", choices=["xgb", "ann"], default=["xgb", "ann"])
    parser.add_argument("--search-iterations", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args(argv)
    for name in ("families", "levels", "models"):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    if args.smoke_test:
        args.n, args.p, args.active, args.reps = 120, 8, 3, 1
        args.families, args.levels = ["exp"], [0.10]
        args.search_iterations, args.cv_folds, args.max_iter = 2, 2, 80
    return args


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)
    validate_config(args, config)
    print(pd.DataFrame(config["grid"]).to_string(index=False), flush=True)
    print("Config SHA256: " + hashlib.sha256(canonical_json(config).encode()).hexdigest(), flush=True)
    if args.validate_only:
        return
    if "xgb" in args.models and not args.export_only:
        from xgboost import XGBRegressor
    run_experiment(args, config)


if __name__ == "__main__":
    main()
