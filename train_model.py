import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from data_processing import load_and_process_data

# Load data
df = load_and_process_data("dataset.csv")

# Encode target
le = LabelEncoder()
df["COPDSEVERITY"] = le.fit_transform(df["COPDSEVERITY"])

# Features
features = [
    'AGE','PackHistory','SmokingRisk',
    'FEV1','FEV1PRED','LungEfficiency','MWT1Best'
]

X = df[features]
y = df["COPDSEVERITY"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=120, max_depth=7)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "models/copd_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("✅ Model saved!")