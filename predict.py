import joblib
import pandas as pd

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")


def predict(reused, gridfins, legs, flights, landing_type="ASDS"):
    input_data = {f: 0 for f in features}
    input_data["reused"] = int(reused)
    input_data["gridfins"] = int(gridfins)
    input_data["legs"] = int(legs)
    input_data["flights"] = int(flights)
    key = f"landing_type_{landing_type}"
    if key in input_data:
        input_data[key] = 1

    df = pd.DataFrame([input_data])
    df_sc = scaler.transform(df)
    pred = model.predict(df_sc)[0]
    prob = model.predict_proba(df_sc)[0][1] if hasattr(model, "predict_proba") else None
    return pred, prob
