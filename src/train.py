import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score

PROCESSED = os.path.join(os.path.dirname(__file__), "../data/processed/")
MODELS    = os.path.join(os.path.dirname(__file__), "../models/")
os.makedirs(MODELS, exist_ok=True)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv(PROCESSED + "domestic_processed.csv")
df.dropna(subset=['EloDiff', 'Form3Home', 'Form3Away',
                  'home_scored_avg', 'away_scored_avg', 'result'], inplace=True)
print(f"Training data: {len(df)} rows\n")

# 7 features now
FEATURES = [
    'EloDiff',
    'Form3Home', 'Form3Away',
    'home_scored_avg', 'home_conceded_avg',
    'away_scored_avg', 'away_conceded_avg'
]

X        = df[FEATURES]
y_result = df['result']
y_over25 = df['over25']
y_btts   = df['btts']

# ─────────────────────────────────────────
# TRAIN + EVALUATE
# ─────────────────────────────────────────
def train_model(X, y, name, task='multiclass'):
    print(f"{'='*50}")
    print(f"MODEL: {name}")
    print(f"{'='*50}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8 if task == 'multiclass' else 6,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation: {cv.mean():.3f} (+/- {cv.std():.3f})")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Test accuracy:    {accuracy_score(y_test, y_pred):.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Feature importances:")
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        bar = '█' * int(imp * 30)
        print(f"  {feat:<22} {imp:.3f} {bar}")

    return model

model_result = train_model(X, y_result, "Match Result (W/D/L)",  task='multiclass')
print()
model_over25 = train_model(X, y_over25, "Over/Under 2.5 Goals",  task='binary')
print()
model_btts   = train_model(X, y_btts,   "Both Teams to Score",   task='binary')

# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────
joblib.dump(model_result, MODELS + "model_result.pkl")
joblib.dump(model_over25, MODELS + "model_over25.pkl")
joblib.dump(model_btts,   MODELS + "model_btts.pkl")
joblib.dump(FEATURES,     MODELS + "features.pkl")

print(f"\n✅ All 3 models saved to /models/")
