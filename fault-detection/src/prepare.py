"""Task 1 - cleaning, split, standardisation (+ feature sets R and Q)."""
from pathlib import Path
import numpy as np
import pandas as pd
from load_data import load_raw

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MEAS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
Q_NAMES = MEAS + [c + "^2" for c in MEAS]


def clean(df):
    """Drop empty columns, rename the label, and coerce measurement columns to numeric."""
    df = df.copy()
    empty = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty).rename(columns={"Output (S)": "fault"})

    invalid_counts = {}
    for col in MEAS:
        original = df[col].copy()
        converted = pd.to_numeric(df[col], errors="coerce")
        invalid = int((converted.isna() & original.notna()).sum())
        if invalid:
            invalid_counts[col] = invalid
        df[col] = converted

    if invalid_counts:
        print("Non-numeric values coerced to NaN:", invalid_counts)
    if df[MEAS].isna().any().any():
        print("Missing values after coercion:\n", df[MEAS].isna().sum().to_string())
    return df, empty


def split_indices(n):
    """1.4: EXACT code given in the assignment."""
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def make_features(df, feature_set):
    """Set R = 6 raw columns. Set Q = raw + squares (squares from RAW values)."""
    X = df[MEAS].to_numpy(dtype=float)
    if feature_set == "Q":
        X = np.hstack([X, X ** 2])
    return X


def get_data(feature_set="R", data_dir=None):
    """Return standardised splits (training mean/std only) + the statistics."""
    df = load_raw(data_dir or ROOT / "data")
    df, _ = clean(df)
    X = make_features(df, feature_set)
    y = df["fault"].to_numpy(dtype=int)
    tr, va, te = split_indices(len(df))

    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)

    def standardize(A):
        return (A - mu) / sd

    return {
        "X_train": standardize(X[tr]), "y_train": y[tr],
        "X_val": standardize(X[va]), "y_val": y[va],
        "X_test": standardize(X[te]), "y_test": y[te],
        "mu": mu, "sd": sd,
    }


def with_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    raw = load_raw(ROOT / "data")
    print("1.1 shape:", raw.shape)
    print("columns:", raw.columns.tolist())
    print(raw.head(5).to_string(), "\n")

    df, removed = clean(raw)
    print("1.2 removed (empty) columns:", removed)
    print("columns now:", df.columns.tolist())
    print("all measurement columns numeric:",
          all(pd.api.types.is_numeric_dtype(df[c]) for c in MEAS))
    print("missing values per column:\n", df.isna().sum().to_string(), "\n")

    n1, n0 = int((df.fault == 1).sum()), int((df.fault == 0).sum())
    print(f"1.3 fault=1: {n1}   fault=0: {n0}\n")

    tr, va, te = split_indices(len(df))
    y = df["fault"].to_numpy()
    split_info = {}
    for name, ix in [("train", tr), ("validation", va), ("test", te)]:
        split_info[name] = (len(ix), float(y[ix].mean()))
        print(f"1.4 {name:10s} rows={len(ix):5d}  fault fraction={y[ix].mean():.4f}")

    d = get_data("R")
    print("\n1.5 train mean after standardising:", np.round(d["X_train"].mean(0), 6))
    print("    train std  after standardising:", np.round(d["X_train"].std(0), 6))

    table1 = {
        "Raw rows / raw columns": f"{raw.shape[0]} / {raw.shape[1]}",
        "Columns removed (names)": ", ".join(removed),
        "Missing values after cleaning": int(df.isna().sum().sum()),
        "Fault rows / no-fault rows": f"{n1} / {n0}",
        "Train / validation / test rows": " / ".join(str(split_info[k][0]) for k in split_info),
        "Fault fraction in train / validation / test":
            " / ".join(f"{split_info[k][1]:.4f}" for k in split_info),
    }
    pd.Series(table1, name="value").to_frame().to_csv(RESULTS / "table1_data_preparation.csv")
    print("\nTable 1 saved.")
