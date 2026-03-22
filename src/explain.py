import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _to_float(x):
    try:
        if isinstance(x, (list, tuple, np.ndarray)):
            x = x[0]

        if isinstance(x, str):
            x = x.replace("[", "").replace("]", "").strip()

        return float(x)
    except Exception:
        return 0.0





def clinical_interpretation(feature, value):

    value = _to_float(value)

    if feature == "HbA1c_level":
        if value > 6.5:
            return "HbA1c is above the diabetic threshold (>6.5)."
        elif value > 5.7:
            return "HbA1c indicates prediabetes."
        else:
            return "HbA1c is within the normal range."

    if feature == "blood_glucose_level":
        if value > 126:
            return "Blood glucose exceeds diabetes diagnostic level."
        elif value > 100:
            return "Blood glucose indicates possible prediabetes."
        else:
            return "Blood glucose is within a healthy range."

    if feature == "bmi":
        if value > 30:
            return "BMI indicates obesity which increases diabetes risk."
        elif value > 25:
            return "BMI indicates overweight."
        else:
            return "BMI is within a healthy range."

    if feature == "age":
        if value > 50:
            return "Diabetes risk increases with age."

    if feature == "hypertension" and value == 1:
        return "Hypertension increases metabolic disease risk."

    if feature == "heart_disease" and value == 1:
        return "Heart disease is associated with metabolic disorders."

    if feature == "smoking_history" and value > 0:
        return "Smoking history contributes to metabolic health risks."

    return ""





def explain_prediction(model, scaled_df, original_df):

    scaled_df = scaled_df.copy().astype(float)
    original_df = original_df.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_df)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    shap_vals = np.array([_to_float(v) for v in shap_vals], dtype=float)

    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = _to_float(base_value[1])
    else:
        base_value = _to_float(base_value)

    explanation = shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=scaled_df.iloc[0].values,
        feature_names=scaled_df.columns.tolist()
    )

    # Waterfall plot
    fig_waterfall = plt.figure()
    shap.plots.waterfall(explanation, show=False)

    # Feature importance plot
    fig_bar = plt.figure()
    shap.plots.bar(explanation, show=False)

    # Build explanation dataframe
    df = pd.DataFrame({
        "feature": scaled_df.columns,
        "value": original_df.iloc[0].values,
        "shap": shap_vals
    })

    df["abs_shap"] = np.abs(df["shap"])
    df = df.sort_values("abs_shap", ascending=False)

    hidden_features = [
        "metabolic_risk",
        "high_risk_comorbidity",
        "age_group"
    ]

    df = df[~df["feature"].isin(hidden_features)]

    increasing = df[df["shap"] > 0].head(4)
    decreasing = df[df["shap"] < 0].head(3)

    feature_names = {
        "HbA1c_level": "HbA1c Level",
        "blood_glucose_level": "Blood Glucose Level",
        "bmi": "Body Mass Index (BMI)",
        "age": "Age",
        "gender": "Gender",
        "smoking_history": "Smoking History",
        "hypertension": "Hypertension",
        "heart_disease": "Heart Disease"
    }

    text = ""

    if len(increasing) > 0:
        text += "###  Factors Increasing Diabetes Risk\n"

        for _, row in increasing.iterrows():
            name = feature_names.get(row["feature"], row["feature"])
            value = round(_to_float(row["value"]), 2)

            interpretation = clinical_interpretation(row["feature"], value)

            text += f"• **{name}**: {value}"

            if interpretation:
                text += f" → {interpretation}"

            text += "\n"

    if len(decreasing) > 0:
        text += "\n###  Factors Reducing Diabetes Risk\n"

        for _, row in decreasing.iterrows():
            name = feature_names.get(row["feature"], row["feature"])
            value = round(_to_float(row["value"]), 2)

            interpretation = clinical_interpretation(row["feature"], value)

            text += f"• **{name}**: {value}"

            if interpretation:
                text += f" → {interpretation}"

            text += "\n"

    return fig_waterfall, fig_bar, text
