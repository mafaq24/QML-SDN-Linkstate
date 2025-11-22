# Minimal QSVM evaluation with alpha tuning on validation, test on hold-out
import json, time, numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

from data_prep_min import load_or_synthetic, stratified_subset, leakage_safe_splits
from hybrid_kernel_min import scale_train_val_test, classical_kernels, quantum_kernels, hybrid

RANDOM_STATE = 42
ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]
C_GRID = [0.1, 1.0, 3.0]
REPS   = 1
CLASS_WEIGHT = None

def fit_eval(Ktr, ytr, Kva, yva, C=1.0, class_weight=None):
    clf = SVC(kernel="precomputed", C=C, class_weight=class_weight)
    t0 = time.time(); clf.fit(Ktr, ytr); t = time.time() - t0
    ypred = clf.predict(Kva)
    yscore = clf.decision_function(Kva)
    return {
        "acc": accuracy_score(yva, ypred),
        "f1": f1_score(yva, ypred, pos_label=1),
        "roc": roc_auc_score(yva, yscore),
        "ap":  average_precision_score(yva, yscore),
        "t": t,
        "clf": clf,
    }

def main():
    # 1) data
    X, y, _ = load_or_synthetic("InSDN_model_ready.csv")
    Xs, ys  = stratified_subset(X, y)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = leakage_safe_splits(Xs, ys)

    # 2) kernels
    Xtr_s, Xva_s, Xte_s = scale_train_val_test(Xtr, Xva, Xte)
    Kc = classical_kernels(Xtr_s, Xva_s, Xte_s)
    Kq = quantum_kernels(Xtr_s, Xva_s, Xte_s, reps=REPS)

    # 3) simple tuning on VAL (alpha, C)
    best = None
    for a in ALPHAS:
        Kh_tr, Kh_va, Kh_te_t = hybrid(Kc, Kq, a)
        for C in C_GRID:
            m = fit_eval(Kh_tr, ytr, Kh_va, yva, C=C, class_weight=CLASS_WEIGHT)
            key = (m["roc"], m["ap"], m["f1"])
            if (best is None) or (key > best["key"]):
                best = {"a": a, "C": C, "key": key, "m": m, "Kh_te_t": Kh_te_t, "Kh_tr": Kh_tr, "ytr": ytr}

    # 4) test with best alpha, C
    clf = SVC(kernel="precomputed", C=best["C"], class_weight=CLASS_WEIGHT)
    clf.fit(best["Kh_tr"], best["ytr"])
    ypred = clf.predict(best["Kh_te_t"])
    yscore = clf.decision_function(best["Kh_te_t"])
    out = {
        "alpha*": best["a"], "C*": best["C"],
        "Accuracy": float(accuracy_score(yte, ypred)),
        "F1":       float(f1_score(yte, ypred, pos_label=1)),
        "ROC_AUC":  float(roc_auc_score(yte, yscore)),
        "AP":       float(average_precision_score(yte, yscore)),
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
