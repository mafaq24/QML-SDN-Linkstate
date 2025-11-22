# Minimal data prep for paper reproduction (leakage-safe)
# Uses: InSDN_model_ready.csv with column 'link_status' (0/1)

import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

RANDOM_STATE = 42
N_SUBSAMPLE  = 100
TEST_SIZE    = 0.20
VAL_SIZE     = 0.25  # of the remaining after test split

def load_or_synthetic(csv_path="InSDN_model_ready.csv"):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        X = df.drop(columns=["link_status"]).values
        y = df["link_status"].values
        return X, y, df.columns.drop("link_status").tolist()
    # Synthetic fallback (same interface)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500, 8))
    y = rng.integers(0, 2, size=500)
    return X, y, [f"f{i}" for i in range(X.shape[1])]

def stratified_subset(X, y, n=N_SUBSAMPLE, seed=RANDOM_STATE):
    Xs, ys = resample(X, y, n_samples=n, stratify=y, random_state=seed)
    return Xs, ys

def leakage_safe_splits(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=RANDOM_STATE):
    Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=val_size, stratify=ytmp, random_state=seed)
    return (Xtr, ytr), (Xva, yva), (Xte, yte)

if __name__ == "__main__":
    X, y, _ = load_or_synthetic()
    Xs, ys = stratified_subset(X, y)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = leakage_safe_splits(Xs, ys)
    print("Shapes:", Xtr.shape, Xva.shape, Xte.shape)
