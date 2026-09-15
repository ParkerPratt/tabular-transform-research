import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import cholesky, eig
from scipy.stats import loguniform
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

OUTPUT_DIR = Path("../results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = OUTPUT_DIR / "holdout_yeo.xlsx"


def make_target_continuous(signal, noise, signal_scale=1.0, intercept=0.0, w=None):
    if w is None:
        w = np.ones(signal.shape[1])
    w = np.asarray(w, dtype=float)
    return intercept + signal_scale * (signal @ w) + noise


def transform_yeojohnson(z, lam):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)

    positive = z >= 0
    negative = ~positive

    if np.isclose(lam, 0.0):
        out[positive] = np.log1p(z[positive])
    else:
        out[positive] = ((z[positive] + 1.0) ** lam - 1.0) / lam

    if np.isclose(lam, 2.0):
        out[negative] = -np.log1p(-z[negative])
    else:
        out[negative] = -(
            (1.0 - z[negative]) ** (2.0 - lam) - 1.0
        ) / (2.0 - lam)

    return out


TRANSFORMS = {
    "yeo": transform_yeojohnson,
}

STRENGTHS = {
    "yeo": [
        1.5286431,
        1.7738810,
        1.9830529,
        2.1798742,
        2.5723859,
    ],
}


def standardize_and_scale(A, means, variances):
    A = np.asarray(A, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)

    sd = A.std(axis=0, ddof=0)
    if np.any(sd == 0):
        raise ValueError("A generated feature has zero sample variance.")

    A_std = (A - A.mean(axis=0)) / sd
    return A_std * np.sqrt(variances) + means


def _nearest_spd(A, eps=1e-9):
    A = (A + A.T) / 2.0
    vals, vecs = eig(A)
    vals = np.real(vals)
    vecs = np.real(vecs)
    vals[vals < eps] = eps
    return vecs @ np.diag(vals) @ vecs.T


def _cholesky_spd(A):
    try:
        return cholesky(A)
    except np.linalg.LinAlgError:
        return cholesky(_nearest_spd(A))


def make_paired_representations(n, means, variances, target_corr, seed, transform_name, strength):
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    target_corr = np.asarray(target_corr, dtype=float)
    p = len(means)

    if means.shape != variances.shape or target_corr.shape != (p, p):
        raise ValueError("Incompatible data-generation parameter dimensions.")

    rng = np.random.default_rng(seed)
    L = _cholesky_spd(target_corr)
    Z_raw = rng.standard_normal(size=(n, p)) @ L.T

    base_representation = standardize_and_scale(Z_raw, means, variances)

    transform_fn = TRANSFORMS[transform_name]
    transformed_raw = transform_fn(Z_raw, strength)
    transformed_representation = standardize_and_scale(
        transformed_raw, means, variances
    )

    return base_representation, transformed_representation


def make_weights(p, n_active):
    n_active = min(n_active, p)
    return np.r_[np.ones(n_active), np.zeros(p - n_active)]


CV = KFold(n_splits=5, shuffle=True, random_state=2025)

PARAM_DISTRIBUTIONS_ANN = {
    "mlp__hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32)],
    "mlp__alpha": loguniform(1e-5, 1e-2),
    "mlp__learning_rate_init": loguniform(1e-4, 1e-2),
    "mlp__batch_size": [32, 64, 128],
}

N_ITER_ANN = 10


def fit_ann(X_train, X_test, y_train, y_test, seed):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=15,
            max_iter=500,
        )),
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DISTRIBUTIONS_ANN,
        n_iter=N_ITER_ANN,
        scoring="neg_mean_squared_error",
        cv=CV,
        n_jobs=24,
        random_state=seed + 99,
        refit=True,
    )
    search.fit(X_train, y_train)
    y_pred = search.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return {
        "mse_test": mse,
        "rmse_test": np.sqrt(mse),
        "r2_test": r2_score(y_test, y_pred),
        "best_params": search.best_params_,
        "cv_best_mse": -search.best_score_,
    }


MODEL_FITTERS = {
    "ann": fit_ann,
}

N_VALUES = [250, 500, 1000, 2000, 5000, 10000]
P = 50
N_ACTIVE = 10
N_REPS = 25
MASTER_SEED = 1337
TARGET_SIGNAL_SCALE = 1.0
TARGET_NOISE_SD = 1.0
TARGET_INTERCEPT = 0.0
TEST_FRACTION = 0.2

CONDITIONS = [
    ("base", "base"),
    ("base", "transformed"),
    ("transformed", "base"),
    ("transformed", "transformed"),
]

KEY_COLUMNS = [
    "n", "p", "n_active", "transform", "strength", "rep",
    "observed_representation", "signal_representation", "model",
]

RESULT_COLUMNS = KEY_COLUMNS + [
    "r2_test", "mse_test", "rmse_test", "cv_best_mse",
    "best_params", "status", "error_message",
]


def load_existing_results(path):
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)

    existing = pd.read_excel(path, sheet_name="results")
    missing = set(KEY_COLUMNS) - set(existing.columns)

    if missing:
        raise ValueError(
            f"Existing workbook incompatible; missing: {sorted(missing)}"
        )

    return existing


def save_results(results, path):
    results = results.sort_values(KEY_COLUMNS).reset_index(drop=True)
    successful = results.loc[results["status"] == "ok"].copy()

    if successful.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            successful.groupby(
                [
                    "n", "p", "n_active", "transform", "strength",
                    "observed_representation", "signal_representation", "model",
                ],
                as_index=False,
            )[["r2_test", "mse_test", "rmse_test"]]
            .agg(["mean", "std"])
        )

        summary.columns = [
            "_".join(str(part) for part in col if str(part))
            if isinstance(col, tuple) else col
            for col in summary.columns
        ]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="results", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"[Progress] Saved {len(results):,} result rows -> {path}")


def key_tuple(record):
    return tuple(record[column] for column in KEY_COLUMNS)


def run_experiment():
    results = load_existing_results(EXCEL_PATH)

    completed_keys = {
        tuple(row)
        for row in results.loc[:, KEY_COLUMNS].itertuples(
            index=False, name=None
        )
    }

    new_rows = []

    means = np.zeros(P)
    variances = np.ones(P)
    target_corr = np.eye(P)
    weights = make_weights(P, N_ACTIVE)

    for n in N_VALUES:
        print(f"\n===== n={n} =====")

        for rep in range(1, N_REPS + 1):
            seed = MASTER_SEED + rep

            split_rng = np.random.default_rng(seed + 101)
            permutation = split_rng.permutation(n)
            test_size = int(np.ceil(TEST_FRACTION * n))
            test_idx = permutation[:test_size]
            train_idx = permutation[test_size:]

            noise_rng = np.random.default_rng(seed + 7)
            response_noise = noise_rng.normal(
                0, TARGET_NOISE_SD, size=n
            )

            for transform_name, strengths in STRENGTHS.items():
                for strength in strengths:
                    print(
                        f"n={n}, rep={rep}/{N_REPS}, "
                        f"transform={transform_name}, "
                        f"strength={strength:g}"
                    )

                    base_rep, transformed_rep = make_paired_representations(
                        n=n,
                        means=means,
                        variances=variances,
                        target_corr=target_corr,
                        seed=seed,
                        transform_name=transform_name,
                        strength=strength,
                    )

                    representations = {
                        "base": base_rep,
                        "transformed": transformed_rep,
                    }

                    targets = {
                        signal_name: make_target_continuous(
                            signal=signal_matrix,
                            noise=response_noise,
                            signal_scale=TARGET_SIGNAL_SCALE,
                            intercept=TARGET_INTERCEPT,
                            w=weights,
                        )
                        for signal_name, signal_matrix in representations.items()
                    }

                    for observed_name, signal_name in CONDITIONS:
                        X = representations[observed_name]
                        y = targets[signal_name]

                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        for model_name, fitter in MODEL_FITTERS.items():
                            key_record = {
                                "n": n,
                                "p": P,
                                "n_active": N_ACTIVE,
                                "transform": transform_name,
                                "strength": strength,
                                "rep": rep,
                                "observed_representation": observed_name,
                                "signal_representation": signal_name,
                                "model": model_name,
                            }

                            if key_tuple(key_record) in completed_keys:
                                continue

                            try:
                                fitted = fitter(
                                    X_train, X_test, y_train, y_test, seed
                                )

                                row = {
                                    **key_record,
                                    "r2_test": fitted["r2_test"],
                                    "mse_test": fitted["mse_test"],
                                    "rmse_test": fitted["rmse_test"],
                                    "cv_best_mse": fitted["cv_best_mse"],
                                    "best_params": json.dumps(
                                        fitted["best_params"],
                                        sort_keys=True,
                                        default=str,
                                    ),
                                    "status": "ok",
                                    "error_message": "",
                                }

                            except Exception as exc:
                                row = {
                                    **key_record,
                                    "r2_test": np.nan,
                                    "mse_test": np.nan,
                                    "rmse_test": np.nan,
                                    "cv_best_mse": np.nan,
                                    "best_params": "",
                                    "status": "failed",
                                    "error_message": str(exc),
                                }

                            new_rows.append(row)
                            completed_keys.add(key_tuple(key_record))

                    if new_rows:
                        results = pd.concat(
                            [results, pd.DataFrame(new_rows)],
                            ignore_index=True,
                        )
                        new_rows.clear()
                        save_results(results, EXCEL_PATH)

    print("Experiment complete.")


run_experiment()
