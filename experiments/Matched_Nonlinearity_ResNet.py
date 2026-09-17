import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import cholesky, eig
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, random_split
except ImportError as e:
    raise ImportError(
        "PyTorch is not installed in this Python environment.\n"
        "Activate your venv, then run:\n"
        "    pip install torch torchvision torchaudio\n"
        "Then re-run this script."
    ) from e


GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

OUTPUT_DIR = Path("../results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = OUTPUT_DIR / "matched_nonlinearity_resnet_n500_p50_a10_fixed.xlsx"


def make_target_continuous(signal, noise, signal_scale=1.0, intercept=0.0, w=None):
    if w is None:
        w = np.ones(signal.shape[1])
    w = np.asarray(w, dtype=float)
    return intercept + signal_scale * (signal @ w) + noise


def transform_identity(z, a):
    return np.asarray(z, dtype=float)


def transform_cubic(z, a):
    z = np.asarray(z, dtype=float)
    return z + a * z**3


def transform_quintic(z, a):
    z = np.asarray(z, dtype=float)
    return z + a * z**5


def transform_exp(z, a):
    z = np.asarray(z, dtype=float)
    if np.isclose(a, 0.0):
        return z
    return np.expm1(a * z) / a


def transform_quad_cubic(z, a):
    z = np.asarray(z, dtype=float)
    return z + a * z**2 + (a**2 / 3.0) * z**3


TRANSFORMS = {
    "cubic": transform_cubic,
    "quintic": transform_quintic,
    "exp": transform_exp,
    "quad": transform_quad_cubic,
}

STRENGTHS = {
    "cubic": [0.1302579, 0.22996560, 0.3532381, 0.5265986, 1.3483315],
    "quintic": [0.0098076, 0.0152675, 0.0204604, 0.0258628, 0.038476],
    "exp": [0.3189425, 0.4551335, 0.5627497, 0.6563857, 0.8218707],
    "quad": [0.1659280, 0.2476384, 0.3223790, 0.3994894, 0.5912237],
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
    transformed_raw = TRANSFORMS[transform_name](Z_raw, strength)
    transformed_representation = standardize_and_scale(transformed_raw, means, variances)

    return base_representation, transformed_representation


def make_weights(p, n_active):
    n_active = min(n_active, p)
    return np.r_[np.ones(n_active), np.zeros(p - n_active)]


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_blocks=3, dropout=0.0):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x).squeeze(-1)


def fit_resnet(X_train, X_test, y_train, y_test, seed):
    hidden_dim = 64
    n_blocks = 3
    dropout = 0.05
    lr = 1e-3
    weight_decay = 1e-4
    batch_size = 64
    epochs = 300
    patience = 30
    validation_fraction = 0.2

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sn_total = len(X_train)
    n_val = max(1, int(n_total * validation_fraction))
    n_train = n_total - n_val

    split_gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(
        range(n_total), [n_train, n_val], generator=split_gen
    )

    train_idx = np.array(train_idx.indices)
    val_idx = np.array(val_idx.indices)

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_train[train_idx])
    X_val_scaled = scaler.transform(X_train[val_idx])
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_fit_scaled, dtype=torch.float32),
        torch.tensor(y_train[train_idx], dtype=torch.float32),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_train[val_idx], dtype=torch.float32),
    )

    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TabularResNet(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_sse = 0.0
        val_n = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)

                val_sse += torch.sum((pred - yb) ** 2).item()
                val_n += yb.numel()

        val_loss = val_sse / val_n

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).detach().cpu().numpy()

    mse = mean_squared_error(y_test, y_pred)
    return {
        "mse_test": mse,
        "rmse_test": np.sqrt(mse),
        "r2_test": r2_score(y_test, y_pred),
        "best_val_mse": best_val_loss,
        "epochs_used": epoch + 1,
    }


N = 500
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
    "r2_test", "mse_test", "rmse_test", "best_val_mse",
    "epochs_used", "status", "error_message",
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

        split_rng = np.random.default_rng(seed + 101)
        permutation = split_rng.permutation(N)
        test_size = int(np.ceil(TEST_FRACTION * N))
        test_idx = permutation[:test_size]
        train_idx = permutation[test_size:]

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

                    key_record = {
                        "n": N,
                        "p": P,
                        "n_active": N_ACTIVE,
                        "transform": transform_name,
                        "strength": strength,
                        "rep": rep,
                        "observed_representation": observed_name,
                        "signal_representation": signal_name,
                        "model": "resnet",
                    }

                    if key_tuple(key_record) in completed_keys:
                        continue

                    try:
                        fitted = fit_resnet(X_train, X_test, y_train, y_test, seed)
                        row = {
                            **key_record,
                            "r2_test": fitted["r2_test"],
                            "mse_test": fitted["mse_test"],
                            "rmse_test": fitted["rmse_test"],
                            "best_val_mse": fitted["best_val_mse"],
                            "epochs_used": fitted["epochs_used"],
                            "status": "ok",
                            "error_message": "",
                        }
                    except Exception as exc:
                        row = {
                            **key_record,
                            "r2_test": np.nan,
                            "mse_test": np.nan,
                            "rmse_test": np.nan,
                            "best_val_mse": np.nan,
                            "epochs_used": np.nan,
                            "status": "failed",
                            "error_message": str(exc),
                        }

                    new_rows.append(row)
                    completed_keys.add(key_tuple(key_record))

                if new_rows:
                    results = pd.concat(
                        [results, pd.DataFrame(new_rows)], ignore_index=True
                    )
                    new_rows.clear()
                    save_results(results, EXCEL_PATH)

    print("Experiment complete.")


run_experiment()
