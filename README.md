# 🎓 Student Exam Score Prediction — ML Production Project

An end-to-end machine learning pipeline that predicts student exam scores based on productivity and lifestyle features. Built with **scikit-learn**, served via **FastAPI**, and containerized with **Docker** — following production-grade ML engineering practices.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML Pipeline                              │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐               │
│  │  Raw CSV ├───►│  train.py    ├───►│  model   │               │
│  │  (data/) │    │  Pipeline +  │    │ .joblib  │               │
│  └──────────┘    │  RF Regressor│    └────┬─────┘               │
│                  └──────────────┘         │                     │
│                                          ▼                     │
│                  ┌──────────────┐    ┌──────────┐    ┌────────┐ │
│                  │  evaluate.py ├───►│  api.py  ├───►│ Docker │ │
│                  │  Metrics     │    │  FastAPI │    │  Image │ │
│                  └──────────────┘    └──────────┘    └────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

```
Client Request                         Response
     │                                      ▲
     ▼                                      │
┌─────────────────────────────────────────────┐
│              FastAPI (api.py)               │
│                                             │
│  Input ──► Pydantic ──► Pipeline ──► JSON   │
│  JSON      Validation   .predict()  Output  │
│                │                            │
│         Schema check                        │
│         against model                       │
└─────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ml-production-project/
├── data/
│   └── student_productivity_dataset.csv   # Raw dataset (5000 samples)
├── models/
│   └── model.joblib                       # Trained sklearn Pipeline
├── src/
│   ├── train.py                           # Training pipeline
│   ├── evaluate.py                        # Model evaluation
│   ├── api.py                             # FastAPI prediction service
│   └── data_processing.py                 # Data utilities
├── tests/
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## 📊 Model Metrics

| Metric | Value |
|--------|-------|
| **MAE**  | 3.9579 |
| **RMSE** | 5.0175 |
| **R²**   | 0.8152 |

**Top Features by Importance:**

| Feature | Importance |
|---------|-----------|
| productivity_score | 0.8230 |
| burnout_level | 0.0494 |
| focus_index | 0.0292 |

---

## 🚀 Getting Started

### 1. Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train.py
```

This will:
- Load and preprocess `data/student_productivity_dataset.csv`
- Split 80/20 train/test
- Train a Random Forest Regressor inside an sklearn Pipeline
- Print evaluation metrics
- Save the pipeline to `models/model.joblib`

### 3. Evaluate the Model

```bash
python src/evaluate.py
```

---

## 🐳 Docker

### Build the Image

```bash
docker build -t ml-student-api .
```

### Run the Container

```bash
docker run -p 8000:8000 ml-student-api
```

The API will be available at `http://localhost:8000`

---

## 📡 API Endpoints

### `GET /` — Root
```bash
curl http://localhost:8000/
```
```json
{"message": "Student Exam Prediction API is running"}
```

### `GET /health` — Health Check
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model_version": "v1"}
```

### `GET /docs` — Interactive API Docs
Open `http://localhost:8000/docs` in your browser for Swagger UI.

### `POST /predict` — Predict Exam Score

**Example Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Example Response:**
```json
{
  "prediction": 30.42,
  "model_version": "v1"
}
```

---

## 🛡️ Production Features

- **sklearn Pipeline** — preprocessing + model as a single object, no train-serve skew
- **Pydantic validation** — strict input types, ranges, and allowed values
- **Startup schema check** — API won't start if model features drift from API schema
- **Health endpoint** — for Docker/k8s readiness probes
- **Non-root Docker user** — container security best practice
- **Logging** — structured request/prediction logging
- **Model versioning** — version tag in every response

---

## 🛠️ Tech Stack

- **ML:** scikit-learn, pandas, numpy
- **API:** FastAPI, Pydantic, Uvicorn
- **Containerization:** Docker
- **Language:** Python 3.10
