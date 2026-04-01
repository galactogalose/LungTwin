import joblib
import pandas as pd
import numpy as np

from nlp_engine import load_medical_pdf, split_docs, create_vector_db, query_db

# Load ML model
model = joblib.load("models/copd_model.pkl")
le = joblib.load("models/label_encoder.pkl")

# Load NLP knowledge base
documents = load_medical_pdf("gold_copd.pdf")
docs = split_docs(documents)
db = create_vector_db(docs)


# Validate input
def validate_input(data):
    required_fields = [
        'AGE',
        'PackHistory',
        'SmokingRisk',
        'FEV1',
        'FEV1PRED',
        'MWT1Best'
    ]

    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing field: {field}")

    # Auto-calculate LungEfficiency
    data['LungEfficiency'] = data['FEV1'] / data['FEV1PRED']

    return data


# Risk score
def calculate_risk_score(model, X):
    probs = model.predict_proba(X)
    return (np.max(probs, axis=1) * 100).round(2)


# Main prediction
def predict_patient(input_data):

        input_data = validate_input(input_data)

        features = [
            'AGE',
            'PackHistory',
            'SmokingRisk',
            'FEV1',
            'FEV1PRED',
            'LungEfficiency',
            'MWT1Best'
        ]

        df_input = pd.DataFrame([input_data])
        df_input = df_input[features]

        pred = model.predict(df_input)[0]
        severity = le.inverse_transform([pred])[0]

        risk = calculate_risk_score(model, df_input)[0]

    # NLP-based advice ONLY
        print("DEBUG → NLP Query Running...")
        query = f"""
Treatment guidelines for {severity} COPD including:
- medications
- oxygen therapy
- pulmonary rehabilitation
"""
        advice = query_db(db, query, severity, risk)
        return severity, risk, advice