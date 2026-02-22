"""
Train a model to predict student exam scores based on productivity features.

Uses sklearn Pipeline + ColumnTransformer so preprocessing + model are
a single serializable object. At inference you feed raw data in — no
manual encoding or scaling needed.

Usage:
    python src/train.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ─── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'student_productivity_dataset.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')
TARGET = 'exam_score'
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ─── 1. Load Dataset ────────────────────────────────────────────────────────
print("=" * 60)
print("1. Loading dataset...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")
print(f"   Target: '{TARGET}' — range [{df[TARGET].min():.2f}, {df[TARGET].max():.2f}]")
print()


# ─── 2. Split Train / Test (on raw data) ────────────────────────────────────
print("=" * 60)
print("2. Splitting train/test...")
print("=" * 60)

df = df.drop(columns=['student_id'])

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"   Train set: {X_train.shape[0]} samples")
print(f"   Test set:  {X_test.shape[0]} samples")
print()


# ─── 3. Build Pipeline ──────────────────────────────────────────────────────
print("=" * 60)
print("3. Building Pipeline (ColumnTransformer + Model)...")
print("=" * 60)

categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()

print(f"   Categorical: {categorical_cols}")
print(f"   Numerical:   {numerical_cols}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

print("   Pipeline built:")
print(f"   {pipeline}")
print()


# ─── 4. Train ───────────────────────────────────────────────────────────────
print("=" * 60)
print("4. Training...")
print("=" * 60)

pipeline.fit(X_train, y_train)
print("   Training complete.")
print()


# ─── 5. Evaluate ────────────────────────────────────────────────────────────
print("=" * 60)
print("5. Evaluation Metrics")
print("=" * 60)

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"   MAE:  {mae:.4f}")
print(f"   RMSE: {rmse:.4f}")
print(f"   R²:   {r2:.4f}")
print()

# Feature importance (top 10)
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
importances = pd.Series(
    pipeline.named_steps['model'].feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("   Top 10 Feature Importances:")
for feat, imp in importances.head(10).items():
    print(f"     {feat:30s} {imp:.4f}")
print()


# ─── 6. Save Pipeline ───────────────────────────────────────────────────────
print("=" * 60)
print("6. Saving pipeline...")
print("=" * 60)

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"   Saved to: {MODEL_PATH}")
print()
print("✅ Done!")
print()
print("   Inference is now just:")
print("     pipeline = joblib.load('models/model.joblib')")
print("     pipeline.predict(raw_dataframe)")
