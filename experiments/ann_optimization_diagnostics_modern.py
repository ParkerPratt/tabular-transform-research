import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import cholesky, eig
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

OUTPUT_DIR = Path("../results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = OUTPUT_DIR / "ann_optimization_diagnostics_modern_mismatch.xlsx"

N = 500
P = 50
N_ACTIVE = 10
N_REPS = 25
MASTER_SEED = 1337

TARGET_SIGNAL_SCALE = 1.0
TARGET_NOISE_SD = 1.0
TARGET_INTERCEPT = 0.0
TEST_FRACTION = 0.2

BASE_ANN_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "alpha": 1e-4,
    "learning_rate_init": 1e-3,
    "batch_size": 64,
    "activation": "relu",
    "solver": "adam",
    "shuffle": True,
}

OPTIMIZATION_BUDGETS = [250, 500, 1000, 2000]

CAPACITY_ARCHITECTURES = {
    "baseline_64_32": (64, 32),
    "wide_128_64": (128, 64),
    "very_wide_256_128": (256, 128),
}
CAPACITY_MAX_ITER = 2000


MISMATCHED_ACTIVE_COUNTS = [1, 5, N_ACTIVE]
ACTIVE_SCALING_MAX_ITER = 2000

RUN_OPTIMIZATION_SWEEP = True
RUN_CAPACITY_SWEEP = True
RUN_MISMATCHED_ACTIVE_SCALING = False
RUN_INACTIVE_CONTROL = False

NONLINEARITY_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]
LEVELS_TO_RUN = [0.20, 0.30]

QUICK_MODE = False
if QUICK_MODE:
    N_REPS = 1
    OPTIMIZATION_BUDGETS = [50, 100]
    CAPACITY_ARCHITECTURES = {"baseline_64_32": (64, 32)}
    CAPACITY_MAX_ITER = 100
    MISMATCHED_ACTIVE_COUNTS = [1, N_ACTIVE]
    ACTIVE_SCALING_MAX_ITER = 100
    LEVELS_TO_RUN = [0.30]


def transform_identity(z, a):
    return np.asarray(z, dtype=float)


def transform_cubic(z, a):
    """T(z) = z + a z^3. Strictly increasing for a >= 0."""
    z = np.asarray(z, dtype=float)
    return z + a * z**3


def transform_quintic(z, a):
    """T(z) = z + a z^5. Strictly increasing for a >= 0."""
    z = np.asarray(z, dtype=float)
    return z + a * z**5


def transform_exp(z, a):
    """T(z) = expm1(a z)/a, with identity at a = 0. Strictly increasing."""
    z = np.asarray(z, dtype=float)
    if np.isclose(a, 0.0):
        return z
    return np.expm1(a * z) / a


def transform_quad_cubic(z, a):
    """T(z) = z + a z^2 + (a^2/3) z^3. Strictly increasing."""
    z = np.asarray(z, dtype=float)
    return z + a * z**2 + (a**2 / 3.0) * z**3


TRANSFORMS = {
    "cubic": transform_cubic,
    "quintic": transform_quintic,
    "exp": transform_exp,
    "quad": transform_quad_cubic,
}

_STRENGTH_LISTS = {
    "cubic": [0.1302579, 0.22996560, 0.3532381, 0.5265986, 1.3483315],
    "quintic": [0.0098076, 0.0152675, 0.0204604, 0.0258628, 0.038476],
    "exp": [0.3189425, 0.4551335, 0.5627497, 0.6563857, 0.8218707],
    "quad": [0.1659280, 0.2476384, 0.3223790, 0.3994894, 0.5912237],
}

STRENGTHS = {
    transform_name: dict(zip(NONLINEARITY_LEVELS, strengths))
    for transform_name, strengths in _STRENGTH_LISTS.items()
}

CONDITIONS = [
    ("base", "base"),
    ("base", "transformed"),
    ("transformed", "base"),
    ("transformed", "transformed"),
]


def condition_code(observed_name, signal_name):
    letter = {"base": "B", "transformed": "T"}
    return letter[observed_name] + letter[signal_name]

def make_target_continuous(
    signal,
    noise,
    signal_scale=1.0,
    intercept=0.0,
    w=None,
):
    if w is None:
        w = np.ones(signal.shape[1])
    w = np.asarray(w, dtype=float)
    return intercept + signal_scale * (signal @ w) + noise


def standardize_and_scale(A, means, variances):
    """Match the marginal standardization convention used in the main experiment."""
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


def make_weights(p, n_active):
    n_active = min(n_active, p)
    return np.r_[np.ones(n_active), np.zeros(p - n_active)]


def make_transform_mask(mode, count=None):
    """
    Return which coordinates receive T.

    mode='all'             : exact layout of the main modern mismatch experiment.
    mode='active_subset'   : only the first `count` active coordinates receive T.
    mode='inactive_only'   : only inactive coordinates receive T.
    """
    mask = np.zeros(P, dtype=bool)

    if mode == "all":
        mask[:] = True
    elif mode == "active_subset":
        if count is None:
            raise ValueError("count is required for active_subset mode.")
        mask[: min(int(count), N_ACTIVE)] = True
    elif mode == "inactive_only":
        mask[N_ACTIVE:] = True
    else:
        raise ValueError(f"Unknown transform-mask mode: {mode}")

    return mask


def make_paired_representations(
    n,
    means,
    variances,
    target_corr,
    seed,
    transform_name,
    strength,
    transform_mask,
):
    """
    Generate base Gaussian B and transformed T from identical Gaussian draws.

    For the main optimization/capacity stages, transform_mask is all True, so this
    reproduces the current BB/BT/TB/TT mismatch layout.
    """
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    target_corr = np.asarray(target_corr, dtype=float)
    transform_mask = np.asarray(transform_mask, dtype=bool)

    p = len(means)
    if means.shape != variances.shape or target_corr.shape != (p, p):
        raise ValueError("Incompatible data-generation parameter dimensions.")
    if transform_mask.shape != (p,):
        raise ValueError("transform_mask must have shape (p,).")

    rng = np.random.default_rng(seed)
    L = _cholesky_spd(target_corr)
    Z_raw = rng.standard_normal(size=(n, p)) @ L.T

    base_representation = standardize_and_scale(Z_raw, means, variances)

    transform_fn = TRANSFORMS[transform_name]
    transformed_raw = Z_raw.copy()
    if np.any(transform_mask):
        transformed_raw[:, transform_mask] = transform_fn(
            Z_raw[:, transform_mask], strength
        )

    transformed_representation = standardize_and_scale(
        transformed_raw, means, variances
    )

    return base_representation, transformed_representation


def paired_data_for_rep(rep, transform_name, strength, transform_mask):
    seed = MASTER_SEED + rep


    split_rng = np.random.default_rng(seed + 101)
    permutation = split_rng.permutation(N)
    test_size = int(np.ceil(TEST_FRACTION * N))
    test_idx = permutation[:test_size]
    train_idx = permutation[test_size:]

    noise_rng = np.random.default_rng(seed + 7)
    response_noise = noise_rng.normal(0, TARGET_NOISE_SD, size=N)

    base_rep, transformed_rep = make_paired_representations(
        n=N,
        means=np.zeros(P),
        variances=np.ones(P),
        target_corr=np.eye(P),
        seed=seed,
        transform_name=transform_name,
        strength=strength,
        transform_mask=transform_mask,
    )

    representations = {
        "base": base_rep,
        "transformed": transformed_rep,
    }

    weights = make_weights(P, N_ACTIVE)
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

    return representations, targets, train_idx, test_idx, seed


def count_network_parameters(model):
    return int(
        sum(w.size for w in model.coefs_)
        + sum(b.size for b in model.intercepts_)
    )


def fit_fixed_ann(
    X_train,
    X_test,
    y_train,
    y_test,
    seed,
    hidden_layer_sizes,
    max_iter,
):
    params = {
        **BASE_ANN_PARAMS,
        "hidden_layer_sizes": tuple(hidden_layer_sizes),
    }

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    **params,
                    random_state=seed,
                    early_stopping=False,
                    max_iter=int(max_iter),
                    tol=1e-6,
                    n_iter_no_change=int(max_iter) + 1,
                ),
            ),
        ]
    )

    convergence_warning = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_train, y_train)
        convergence_warning = any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )

    mlp = pipeline.named_steps["mlp"]
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    mse_train = mean_squared_error(y_train, pred_train)
    mse_test = mean_squared_error(y_test, pred_test)

    return {
        "r2_train": r2_score(y_train, pred_train),
        "mse_train": mse_train,
        "rmse_train": np.sqrt(mse_train),
        "r2_test": r2_score(y_test, pred_test),
        "mse_test": mse_test,
        "rmse_test": np.sqrt(mse_test),
        "generalization_gap_mse": mse_test - mse_train,
        "n_iter_fitted": int(mlp.n_iter_),
        "final_training_loss": float(mlp.loss_),
        "convergence_warning": bool(convergence_warning),
        "n_network_parameters": count_network_parameters(mlp),
        "loss_curve": [float(v) for v in mlp.loss_curve_],
        "ann_params": {
            **params,
            "early_stopping": False,
            "max_iter": int(max_iter),
            "tol": 1e-6,
            "n_iter_no_change": int(max_iter) + 1,
        },
    }


RESULT_COLUMNS = [
    "stage",
    "n",
    "p",
    "n_active",
    "rep",
    "transform",
    "nonlinearity_level",
    "strength",
    "transform_scenario",
    "n_coordinates_transformed",
    "n_active_transformed",
    "n_inactive_transformed",
    "observed_representation",
    "signal_representation",
    "condition",
    "architecture_name",
    "hidden_layer_sizes",
    "max_iter",
    "r2_train",
    "mse_train",
    "rmse_train",
    "r2_test",
    "mse_test",
    "rmse_test",
    "generalization_gap_mse",
    "n_iter_fitted",
    "final_training_loss",
    "convergence_warning",
    "n_network_parameters",
    "loss_curve",
    "ann_params",
    "status",
    "error_message",
]


def run_configuration(
    stage,
    transform_name,
    nonlinearity_level,
    strength,
    transform_scenario,
    transform_mask,
    architecture_name,
    hidden_layer_sizes,
    max_iter,
):
    transform_mask = np.asarray(transform_mask, dtype=bool)
    n_active_transformed = int(transform_mask[:N_ACTIVE].sum())
    n_inactive_transformed = int(transform_mask[N_ACTIVE:].sum())

    rows = []
    for rep in range(1, N_REPS + 1):
        representations, targets, train_idx, test_idx, seed = paired_data_for_rep(
            rep=rep,
            transform_name=transform_name,
            strength=strength,
            transform_mask=transform_mask,
        )

        for observed_name, signal_name in CONDITIONS:
            X = representations[observed_name]
            y = targets[signal_name]

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            base = {
                "stage": stage,
                "n": N,
                "p": P,
                "n_active": N_ACTIVE,
                "rep": rep,
                "transform": transform_name,
                "nonlinearity_level": float(nonlinearity_level),
                "strength": float(strength),
                "transform_scenario": transform_scenario,
                "n_coordinates_transformed": int(transform_mask.sum()),
                "n_active_transformed": n_active_transformed,
                "n_inactive_transformed": n_inactive_transformed,
                "observed_representation": observed_name,
                "signal_representation": signal_name,
                "condition": condition_code(observed_name, signal_name),
                "architecture_name": architecture_name,
                "hidden_layer_sizes": json.dumps(list(hidden_layer_sizes)),
                "max_iter": int(max_iter),
            }

            try:
                fitted = fit_fixed_ann(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    seed=seed,
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=max_iter,
                )

                row = {
                    **base,
                    **{
                        k: v
                        for k, v in fitted.items()
                        if k not in {"loss_curve", "ann_params"}
                    },
                    "loss_curve": json.dumps(fitted["loss_curve"]),
                    "ann_params": json.dumps(
                        fitted["ann_params"], sort_keys=True, default=str
                    ),
                    "status": "ok",
                    "error_message": "",
                }
            except Exception as exc:
                row = {
                    **base,
                    **{
                        col: np.nan
                        for col in [
                            "r2_train",
                            "mse_train",
                            "rmse_train",
                            "r2_test",
                            "mse_test",
                            "rmse_test",
                            "generalization_gap_mse",
                            "n_iter_fitted",
                            "final_training_loss",
                            "n_network_parameters",
                        ]
                    },
                    "convergence_warning": np.nan,
                    "loss_curve": "",
                    "ann_params": "",
                    "status": "failed",
                    "error_message": str(exc),
                }

            rows.append(row)

        print(
            f"[{stage}] transform={transform_name} | L={nonlinearity_level:.2f} | "
            f"{transform_scenario} | {architecture_name} | max_iter={max_iter} | "
            f"rep={rep}/{N_REPS}"
        )

    return rows


def iter_transform_settings():
    for transform_name in TRANSFORMS:
        for level in LEVELS_TO_RUN:
            if level not in STRENGTHS[transform_name]:
                raise ValueError(
                    f"No strength for transform={transform_name}, nonlinearity={level}."
                )
            yield transform_name, level, STRENGTHS[transform_name][level]


def run_experiment():
    all_rows = []

    if RUN_OPTIMIZATION_SWEEP:
        transform_mask = make_transform_mask("all")
        for transform_name, level, strength in iter_transform_settings():
            for budget in OPTIMIZATION_BUDGETS:
                all_rows.extend(
                    run_configuration(
                        stage="optimization_budget",
                        transform_name=transform_name,
                        nonlinearity_level=level,
                        strength=strength,
                        transform_scenario="all_predictors",
                        transform_mask=transform_mask,
                        architecture_name="baseline_64_32",
                        hidden_layer_sizes=BASE_ANN_PARAMS["hidden_layer_sizes"],
                        max_iter=budget,
                    )
                )

    if RUN_CAPACITY_SWEEP:
        transform_mask = make_transform_mask("all")
        for transform_name, level, strength in iter_transform_settings():
            for architecture_name, hidden_sizes in CAPACITY_ARCHITECTURES.items():
                all_rows.extend(
                    run_configuration(
                        stage="capacity",
                        transform_name=transform_name,
                        nonlinearity_level=level,
                        strength=strength,
                        transform_scenario="all_predictors",
                        transform_mask=transform_mask,
                        architecture_name=architecture_name,
                        hidden_layer_sizes=hidden_sizes,
                        max_iter=CAPACITY_MAX_ITER,
                    )
                )

    if RUN_MISMATCHED_ACTIVE_SCALING:
        for transform_name, level, strength in iter_transform_settings():
            for count in MISMATCHED_ACTIVE_COUNTS:
                transform_mask = make_transform_mask("active_subset", count=count)
                all_rows.extend(
                    run_configuration(
                        stage="mismatched_active_scaling",
                        transform_name=transform_name,
                        nonlinearity_level=level,
                        strength=strength,
                        transform_scenario=f"active_subset_{count}",
                        transform_mask=transform_mask,
                        architecture_name="baseline_64_32",
                        hidden_layer_sizes=BASE_ANN_PARAMS["hidden_layer_sizes"],
                        max_iter=ACTIVE_SCALING_MAX_ITER,
                    )
                )

    if RUN_INACTIVE_CONTROL:
        transform_mask = make_transform_mask("inactive_only")
        for transform_name, level, strength in iter_transform_settings():
            all_rows.extend(
                run_configuration(
                    stage="inactive_control",
                    transform_name=transform_name,
                    nonlinearity_level=level,
                    strength=strength,
                    transform_scenario="inactive_only",
                    transform_mask=transform_mask,
                    architecture_name="baseline_64_32",
                    hidden_layer_sizes=BASE_ANN_PARAMS["hidden_layer_sizes"],
                    max_iter=ACTIVE_SCALING_MAX_ITER,
                )
            )

    results = pd.DataFrame(all_rows, columns=RESULT_COLUMNS)
    save_results(results)


def save_results(results):
    group_columns = [
        "stage",
        "transform",
        "nonlinearity_level",
        "strength",
        "transform_scenario",
        "n_coordinates_transformed",
        "n_active_transformed",
        "n_inactive_transformed",
        "observed_representation",
        "signal_representation",
        "condition",
        "architecture_name",
        "hidden_layer_sizes",
        "max_iter",
    ]

    metric_columns = [
        "r2_train",
        "r2_test",
        "mse_train",
        "mse_test",
        "generalization_gap_mse",
        "n_iter_fitted",
        "final_training_loss",
        "n_network_parameters",
    ]

    successful = results.loc[results["status"] == "ok"].copy()

    if successful.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            successful.groupby(group_columns, as_index=False)[metric_columns]
            .agg(["mean", "std"])
        )
        summary.columns = [
            "_".join(str(part) for part in col if str(part))
            if isinstance(col, tuple)
            else col
            for col in summary.columns
        ]
        summary = summary.reset_index(drop=True)

    loss_curves_long, mean_loss_curves = aggregate_loss_curves(
        successful, group_columns
    )

    config_rows = [
        ("N", N),
        ("P", P),
        ("N_ACTIVE", N_ACTIVE),
        ("N_REPS", N_REPS),
        ("MASTER_SEED", MASTER_SEED),
        ("TARGET_NOISE_SD", TARGET_NOISE_SD),
        ("TEST_FRACTION", TEST_FRACTION),
        ("BASE_ANN_PARAMS", json.dumps(BASE_ANN_PARAMS, default=str)),
        ("OPTIMIZATION_BUDGETS", json.dumps(OPTIMIZATION_BUDGETS)),
        ("CAPACITY_ARCHITECTURES", json.dumps(CAPACITY_ARCHITECTURES)),
        ("CAPACITY_MAX_ITER", CAPACITY_MAX_ITER),
        ("NONLINEARITY_LEVELS", json.dumps(NONLINEARITY_LEVELS)),
        ("LEVELS_TO_RUN", json.dumps(LEVELS_TO_RUN)),
        ("STRENGTHS", json.dumps(STRENGTHS)),
        ("MISMATCHED_ACTIVE_COUNTS", json.dumps(MISMATCHED_ACTIVE_COUNTS)),
        ("ACTIVE_SCALING_MAX_ITER", ACTIVE_SCALING_MAX_ITER),
        ("RUN_OPTIMIZATION_SWEEP", RUN_OPTIMIZATION_SWEEP),
        ("RUN_CAPACITY_SWEEP", RUN_CAPACITY_SWEEP),
        ("RUN_MISMATCHED_ACTIVE_SCALING", RUN_MISMATCHED_ACTIVE_SCALING),
        ("RUN_INACTIVE_CONTROL", RUN_INACTIVE_CONTROL),
    ]
    config_df = pd.DataFrame(config_rows, columns=["parameter", "value"])

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="raw_results", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        mean_loss_curves.to_excel(writer, sheet_name="mean_loss_curves", index=False)
        config_df.to_excel(writer, sheet_name="configuration", index=False)

    print(f"Saved {len(results):,} result rows to {EXCEL_PATH}")
    print(f"Successful fits: {(results['status'] == 'ok').sum():,}")
    print(f"Failed fits: {(results['status'] == 'failed').sum():,}")


def aggregate_loss_curves(frame, group_columns):
    curve_rows = []

    for _, row in frame.iterrows():
        if not row["loss_curve"]:
            continue
        curve = json.loads(row["loss_curve"])
        for epoch, loss in enumerate(curve, start=1):
            curve_rows.append(
                {
                    **{col: row[col] for col in group_columns},
                    "rep": row["rep"],
                    "epoch": epoch,
                    "training_loss": loss,
                }
            )

    if not curve_rows:
        return pd.DataFrame(), pd.DataFrame()

    curves_long = pd.DataFrame(curve_rows)
    mean_curves = (
        curves_long.groupby(group_columns + ["epoch"], as_index=False)
        ["training_loss"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    return curves_long, mean_curves


if __name__ == "__main__":
    run_experiment()
