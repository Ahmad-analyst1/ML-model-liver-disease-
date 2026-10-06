# Liver Disease Screening – deployable model (from IJCRT2607605)

1. `pip install -r requirements.txt`
2. Download ILPD (UCI #225) into `data/`, then `python train.py --data data/<file>.csv`
   -> writes `model/model.joblib` + `model/metrics.json`
3. API: `uvicorn app.main:app --port 8000` (docs at /docs) | UI: `streamlit run streamlit_app.py`
4. Docker: `docker build -t liver-screen . && docker run -p 8000:8000 liver-screen`

Example: `curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"age":50,"gender":1,"total_bilirubin":3,"direct_bilirubin":1.2,"alk_phosphatase":300,"alt":120,"ast":150,"total_proteins":6,"albumin":2.8}'`

Not a diagnostic device. ILPD = general liver disease, not NAFLD-specific.
