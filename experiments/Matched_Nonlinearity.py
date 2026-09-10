import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import cholesky, eig
from scipy.stats import loguniform, randint
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

OUTPUT_DIR = Path("../results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = OUTPUT_DIR / "matched_nonlinearity_transform_mismatch_n10000_p50_a10.xlsx"


def make_target_continuous(signal, noise, signal_scale=1.0, intercept=0.0, w=None):
    if w is None:
        w = np.ones(signal.shape[1])
    w = np.asarray(w, dtype=float)
    return intercept + signal_scale * (signal @ w) + noise


def transform_identity(z, a):
    return np.asarray(z, dtype=float)


def transform_cubic(z, a):
    """T(z)=z+a z^3. Strictly increasing for a >= 0."""
    z = np.asarray(z, dtype=float)
    return z + a * z**3

def transform_quintic(z,a):
    z = np.asarray(z, dtype=float)
    return z + a * z**5

def transform_exp(z, a):
    """T(z)=expm1(a z)/a, with identity at a=0. Strictly increasing."""
    z = np.asarray(z, dtype=float)
    if np.isclose(a, 0.0):
        return z
    return np.expm1(a * z) / a


def transform_quad_cubic(z, a):
    """
    T_a(z) = z + a z^2 + (a^2/3) z^3
    """
    z = np.asarray(z, dtype=float)
    return z + a * z**2 + (a**2 / 3.0) * z**3

TRANSFORMS = {
    "cubic": transform_cubic,
    "quintic": transform_quintic,
    "exp": transform_exp,
    "quad": transform_quad_cubic,
}

# Strengths are family-specific; they are NOT intended to be numerically comparable.
STRENGTHS = {
    "cubic": [0.1302579, 0.22996560, 0.3532381, 0.5265986, 1.3483315],
    "quintic":  [0.0098076, 0.0152675, 0.0204604, 0.0258628, 0.038476],
    "exp":   [0.3189425, 0.4551335, 0.5627497, 0.6563857, 0.8218707],
    "quad":  [0.1659280, 0.2476384, 0.3223790, 0.3994894, 0.5912237],
}


def standardize_and_scale(A, means, variances):
    """Give each sample marginal exactly the requested sample mean/variance."""
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
    """Generate base Gaussian and transformed representations from identical Z draws."""
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
    transformed_representation = standardize_and_scale(transformed_raw, means, variances)

    return base_representation, transformed_representation


def make_weights(p, n_active):
    n_active = min(n_active, p)
    return np.r_[np.ones(n_active), np.zeros(p - n_active)]


CV = KFold(n_splits=5, shuffle=True, random_state=2025)

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

N_ITER_XGB = 10
N_ITER_ANN = 10


def fit_xgb(X_train, X_test, y_train, y_test, seed):
    search = RandomizedSearchCV(
        estimator=XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=2020,
            n_jobs=1,
            verbosity=0,
        ),
        param_distributions=PARAM_DISTRIBUTIONS_XGB,
        n_iter=N_ITER_XGB,
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


MODEL_FITTERS = {"xgb": fit_xgb, "ann": fit_ann}

N = 10000
P = 50
N_ACTIVE = 10
N_REPS = 25
MASTER_SEED = 1337
TARGET_SIGNAL_SCALE = 1.0
TARGET_NOISE_SD = 1.0
TARGET_INTERCEPT = 0.0
TEST_FRACTION = 0.2

# observed = representation supplied to model
# signal   = representation used to generate y
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
        raise ValueError(f"Existing workbook incompatible; missing: {sorted(missing)}")
    return existing


def save_results(results, path):
    results = results.sort_values(KEY_COLUMNS).reset_index(drop=True)
    successful = results.loc[results["status"] == "ok"].copy()

    if successful.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            successful.groupby(
                ["n", "p", "n_active", "transform", "strength",
                 "observed_representation", "signal_representation", "model"],
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
        for row in results.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)
    }
    new_rows = []

    means = np.zeros(P)
    variances = np.ones(P)
    target_corr = np.eye(P)
    weights = make_weights(P, N_ACTIVE)

    for rep in range(1, N_REPS + 1):
        seed = MASTER_SEED + rep

        # Same train/test indices for every family, strength, and condition.
        split_rng = np.random.default_rng(seed + 101)
        permutation = split_rng.permutation(N)
        test_size = int(np.ceil(TEST_FRACTION * N))
        test_idx = permutation[:test_size]
        train_idx = permutation[test_size:]

        # Same response noise everywhere in a replication.
        noise_rng = np.random.default_rng(seed + 7)
        response_noise = noise_rng.normal(0, TARGET_NOISE_SD, size=N)

        for transform_name, strengths in STRENGTHS.items():
            for strength in strengths:
                print(
                    f"rep={rep}/{N_REPS}, transform={transform_name}, "
                    f"strength={strength:g}"
                )

                base_rep, transformed_rep = make_paired_representations(
                    n=N,
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
                            "n": N,
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
                            fitted = fitter(X_train, X_test, y_train, y_test, seed)
                            row = {
                                **key_record,
                                "r2_test": fitted["r2_test"],
                                "mse_test": fitted["mse_test"],
                                "rmse_test": fitted["rmse_test"],
                                "cv_best_mse": fitted["cv_best_mse"],
                                "best_params": json.dumps(
                                    fitted["best_params"], sort_keys=True, default=str
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

                # Save after each transform-strength pair.
                if new_rows:
                    results = pd.concat([results, pd.DataFrame(new_rows)], ignore_index=True)
                    new_rows.clear()
                    save_results(results, EXCEL_PATH)

    print("Experiment complete.")


run_experiment()
