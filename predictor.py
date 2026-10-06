import joblib, os
import pandas as pd

_BUNDLE = None
def bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(os.getenv("MODEL_PATH", "model/model.joblib"))
    return _BUNDLE

def predict(record: dict) -> dict:
    b = bundle()
    prob = float(b["pipeline"].predict_proba(pd.DataFrame([record])[b["features"]])[0, 1])
    thr = b["threshold"]
    band = "high" if prob >= max(thr, 0.8) else ("elevated" if prob >= thr else "low")
    return {"probability": round(prob, 4), "threshold": thr, "flag": prob >= thr, "risk_band": band,
            "disclaimer": "Screening aid only; not a diagnosis. Confirm with a clinician (ultrasound/FibroScan, labs)."}
