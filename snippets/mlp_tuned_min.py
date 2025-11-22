"""
Minimal, leakage-safe MLP baseline (reproducible snippet)

- Subsample (stratified) for a quick CPU run
- Split: TRAIN / VAL / TEST
- Tune on VAL (no leakage), then retrain on TRAIN+VAL
- Evaluate on TEST and print metrics as JSON
- Uses real CSV if present; otherwise generates synthetic fallback

Expected CSV: InSDN_model_ready.csv with target column 'link_status'
"""

import os, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# TensorFlow (CPU)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------ Config (small & fast) ------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

N_SUBSAMPLE = 5000       # keep small for reviewers
TEST_SIZE   = 0.20
VAL_SIZE    = 0.25       # of remaining after test
CSV_PATH    = "InSDN_model_ready.csv"

# ------------------ Data helpers ---------------------------
def load_or_synthetic(csv_path=CSV_PATH):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path).reset_index(drop=True)
        X = df.drop(columns=["link_status"]).to_numpy()
        y = df["link_status"].to_numpy()
        return X, y
    # fallback (same interface)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(6000, 16))
    y = rng.integers(0, 2, size=6000)
    return X, y

def leakage_safe_splits(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=RANDOM_STATE):
    Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    Xtr, Xva, ytr, yva  = train_test_split(Xtmp, ytmp, test_size=val_size, stratify=ytmp, random_state=seed)
    return (Xtr, ytr), (Xva, yva), (Xte, yte)

# ------------------ Model -------------------------------
def build_mlp(input_dim, n_hidden=2, width=128, dropout=0.2, lr=1e-3):
    model = keras.Sequential([layers.Input(shape=(input_dim,))])
    for _ in range(n_hidden):
        model.add(layers.Dense(width, activation="relu"))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc", curve="ROC")],
    )
    return model

def tune_on_val(Xtr, ytr, Xva, yva, max_epochs=60):
    es = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=8, restore_best_weights=True, verbose=0
    )
    # small, sensible search
    grid = [
        dict(n_hidden=1, width=128, dropout=0.2, lr=1e-3, batch=128),
        dict(n_hidden=2, width=128, dropout=0.3, lr=1e-3, batch=128),
        dict(n_hidden=2, width=256, dropout=0.2, lr=1e-3, batch=128),
        dict(n_hidden=3, width=256, dropout=0.3, lr=5e-4, batch=128),
    ]
    best = None
    for cfg in grid:
        m = build_mlp(Xtr.shape[1], **{k: cfg[k] for k in ["n_hidden","width","dropout","lr"]})
        h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=max_epochs,
                  batch_size=cfg["batch"], verbose=0, callbacks=[es])
        val_auc = float(np.max(h.history["val_auc"]))
        if (best is None) or (val_auc > best["val_auc"]):
            best = dict(cfg=cfg, val_auc=val_auc, model=m)
    return best

# ------------------ Main -------------------------------
def main():
    # 1) data
    X, y = load_or_synthetic()
    Xs, ys = resample(X, y, n_samples=min(N_SUBSAMPLE, len(y)), stratify=y, random_state=RANDOM_STATE)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = leakage_safe_splits(Xs, ys)

    # 2) scale (fit on TRAIN only)
    sc = MinMaxScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)

    # 3) tune on VAL
    best = tune_on_val(Xtr_s, ytr, Xva_s, yva)

    # 4) retrain on TRAIN+VAL; test once
    Xtv = np.vstack([Xtr_s, Xva_s]); ytv = np.concatenate([ytr, yva])
    final = build_mlp(Xtv.shape[1], **{k: best["cfg"][k] for k in ["n_hidden","width","dropout","lr"]})
    es = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True, verbose=0)
    t0 = time.time()
    final.fit(Xtv, ytv, validation_split=0.1, epochs=80, batch_size=best["cfg"]["batch"], verbose=0, callbacks=[es])
    train_time = time.time() - t0

    y_prob = final.predict(Xte_s, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    out = {
        "Model": f"MLP tuned (n_h={best['cfg']['n_hidden']}, w={best['cfg']['width']})",
        "Accuracy": float(accuracy_score(yte, y_pred)),
        "F1":       float(f1_score(yte, y_pred, pos_label=1)),
        "ROC_AUC":  float(roc_auc_score(yte, y_prob)),
        "AP":       float(average_precision_score(yte, y_prob)),
        "Train_Time_s": float(train_time),
        "Val_AUC*": float(best["val_auc"]),
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
