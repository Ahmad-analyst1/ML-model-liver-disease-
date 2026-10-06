from fastapi import FastAPI
from pydantic import BaseModel, Field
from .predictor import predict, bundle

app = FastAPI(title="Liver Disease Screening API", version="1.0")

class Patient(BaseModel):
    age: float = Field(ge=1, le=120)
    gender: int = Field(ge=0, le=1, description="1=male, 0=female")
    total_bilirubin: float = Field(ge=0, le=100, description="mg/dL")
    direct_bilirubin: float = Field(ge=0, le=50, description="mg/dL")
    alk_phosphatase: float = Field(ge=10, le=5000, description="IU/L")
    alt: float = Field(ge=1, le=5000, description="IU/L")
    ast: float = Field(ge=1, le=10000, description="IU/L")
    total_proteins: float = Field(ge=1, le=15, description="g/dL")
    albumin: float = Field(ge=0.5, le=8, description="g/dL")
    ag_ratio: float | None = Field(default=None, ge=0.1, le=5, description="optional; median-imputed")

@app.on_event("startup")
def _load(): bundle()

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/predict")
def do_predict(p: Patient): return predict(p.model_dump())
