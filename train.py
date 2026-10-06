"""Train the liver-disease screening model described in the paper (Logistic Regression on ILPD).

Usage:
    python train.py --data data/Indian_Liver_Patient_Dataset.csv
Get ILPD from UCI: https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
Accepts the raw UCI file (no header) or a headered CSV (Kaggle version).
"""
import argparse, json
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "gender", "total_bilirubin", "direct_bilirubin", "alk_phosphatase",
            "alt", "ast", "total_proteins", "albumin", "ag_ratio"]
RAW_NAMES = FEATURES + ["target"]


def load(path):
    df = pd.read_csv(path, header=None)
    if not str(df.iloc[0, 0]).strip().replace(".", "").isdigit():  # headered file
        df = pd.read_csv(path)
    df = df.iloc[:, :11].copy()
    df.columns = RAW_NAMES
    df["gender"] = df["gender"].astype(str).str.strip().str.lower().map(
        {"male": 1, "m": 1, "female": 0, "f": 0})
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # ILPD: 1 = liver patient, 2 = non-patient  ->  1 / 0
    df["target"] = (pd.to_numeric(df["target"]) == 1).astype(int)
    return df.drop_duplicates() if False else df  # keep as in paper (583 rows)


def build_pipeline():
    # Imputer + scaler live INSIDE the pipeline so they're fit on training folds only (no leakage).
    pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                               ("sc", StandardScaler())]), FEATURES)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, random_state=42))])


def metrics(y, prob, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return dict(threshold=round(float(thr), 4), accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0), sensitivity=recall_score(y, pred),
                specificity=tn / (tn + fp), f1=f1_score(y, pred), auc=roc_auc_score(y, prob),
                tp=int(tp), fn=int(fn), tn=int(tn), fp=int(fp))


def pick_threshold(y, prob, min_sens):
    """Highest-specificity threshold that still keeps sensitivity >= min_sens."""
    best = 0.5
    for t in np.linspace(0.05, 0.95, 181):
        pred = (prob >= t).astype(int)
        if recall_score(y, pred) >= min_sens:
            best = t  # thresholds increase -> specificity increases while constraint holds
    return float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--min-sensitivity", type=float, default=0.90)
    ap.add_argument("--out", default="model")
    a = ap.parse_args()

    df = load(a.data)
    X, y = df[FEATURES], df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Threshold chosen from out-of-fold predictions on TRAIN only (test set stays untouched).
    cv = StratifiedKFold(10, shuffle=True, random_state=42)
    oof = cross_val_predict(build_pipeline(), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    thr = pick_threshold(ytr, oof, a.min_sensitivity)

    model = build_pipeline().fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    report = {"holdout_default_0.5": metrics(yte, prob, 0.5),
              "holdout_tuned_threshold": metrics(yte, prob, thr),
              "cv_oof_auc_train": roc_auc_score(ytr, oof), "n_train": len(Xtr), "n_test": len(Xte)}

    final = build_pipeline().fit(X, y)  # refit on all data for deployment
    coefs = dict(zip(FEATURES, final.named_steps["clf"].coef_[0].round(4).tolist()))
    joblib.dump({"pipeline": final, "features": FEATURES, "threshold": thr}, f"{a.out}/model.joblib")
    json.dump({**report, "coefficients_standardized": coefs}, open(f"{a.out}/metrics.json", "w"),
              indent=2, default=float)
    print(json.dumps(report, indent=2, default=float)); print("coefficients:", coefs)


if __name__ == "__main__":
    main()
