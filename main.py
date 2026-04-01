from predictor import predict_patient

# Sample patient
patient = {
    'AGE': 65,
    'PackHistory': 40,
    'SmokingRisk': 2,
    'FEV1': 1.2,
    'FEV1PRED': 2.5,
    'MWT1Best': 300
}

severity, risk, advice = predict_patient(patient)

print("\n=== COPD AI RESULT ===")
print("Severity:", severity)
print("Risk Score:", risk)

print("\n📚 Medical Advice (NLP ONLY):")
print(advice)