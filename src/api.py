from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import logging
import os

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_VERSION = "v1"
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.joblib')

# ─── Load Pipeline ───────────────────────────────────────────────────────────
try:
    pipeline = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


# ─── Input Validation ────────────────────────────────────────────────────────
class StudentInput(BaseModel):
    age: int = Field(..., ge=14, le=30, description="Student age")
    gender: Literal["Male", "Female", "Other"] = Field(..., description="Student gender")
    academic_level: Literal["High School", "Undergraduate", "Postgraduate"] = Field(..., description="Academic level")
    study_hours: float = Field(..., ge=0, le=24)
    self_study_hours: float = Field(..., ge=0, le=24)
    online_classes_hours: float = Field(..., ge=0, le=24)
    social_media_hours: float = Field(..., ge=0, le=24)
    gaming_hours: float = Field(..., ge=0, le=24)
    sleep_hours: float = Field(..., ge=0, le=24)
    screen_time_hours: float = Field(..., ge=0, le=24)
    exercise_minutes: int = Field(..., ge=0, le=300)
    caffeine_intake_mg: int = Field(..., ge=0, le=1000)
    part_time_job: Literal[0, 1] = Field(..., description="0 = No, 1 = Yes")
    upcoming_deadline: Literal[0, 1] = Field(..., description="0 = No, 1 = Yes")
    internet_quality: Literal["Poor", "Average", "Good"] = Field(..., description="Internet quality")
    mental_health_score: int = Field(..., ge=1, le=10)
    focus_index: float = Field(..., ge=0, le=100)
    burnout_level: float = Field(..., ge=0, le=100)
    productivity_score: float = Field(..., ge=0, le=100)

    model_config = {"json_schema_extra": {
        "example": {
            "age": 20,
            "gender": "Male",
            "academic_level": "Undergraduate",
            "study_hours": 5.5,
            "self_study_hours": 2.0,
            "online_classes_hours": 1.5,
            "social_media_hours": 2.0,
            "gaming_hours": 1.0,
            "sleep_hours": 7.0,
            "screen_time_hours": 6.0,
            "exercise_minutes": 60,
            "caffeine_intake_mg": 150,
            "part_time_job": 0,
            "upcoming_deadline": 1,
            "internet_quality": "Good",
            "mental_health_score": 7,
            "focus_index": 35.0,
            "burnout_level": 40.0,
            "productivity_score": 55.0
        }
    }}


# ─── Startup Schema Validation ──────────────────────────────────────────────
pipeline_features = set(pipeline.feature_names_in_)
schema_features = set(StudentInput.model_fields.keys())

if pipeline_features != schema_features:
    missing_in_schema = pipeline_features - schema_features
    extra_in_schema = schema_features - pipeline_features
    raise RuntimeError(
        f"Schema mismatch!\n"
        f"  Missing in Pydantic schema: {missing_in_schema or 'none'}\n"
        f"  Extra in Pydantic schema:   {extra_in_schema or 'none'}"
    )

logger.info(f"Schema validated: {len(pipeline_features)} features match")


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Student Exam Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(input_data: StudentInput):
    logger.info("Prediction requested")

    df = pd.DataFrame([input_data.model_dump()])

    # Pipeline handles encoding + prediction internally
    prediction = pipeline.predict(df)

    logger.info(f"Prediction result: {prediction[0]:.2f}")

    return {
        "prediction": float(prediction[0]),
        "model_version": MODEL_VERSION
    }
