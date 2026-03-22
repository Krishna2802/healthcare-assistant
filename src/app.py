import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

from explain import explain_prediction
from llm_rag import ask_llm



# Loading model 


model = xgb.XGBClassifier()
model.load_model("models/xgb_model.json")

robust_scaler = joblib.load("models/robust_scaler.joblib")
std_scaler = joblib.load("models/std_scaler.joblib")

le_gender = joblib.load("models/le_gender.joblib")
le_smoking = joblib.load("models/le_smoking.joblib")

feature_columns = joblib.load("models/feature_columns.joblib")



# Streamlit page


st.set_page_config(page_title="Healthcare AI Assistant", layout="wide")

st.title("🩺 AI Healthcare Assistant")
st.write("Predict diabetes risk and ask health questions.")




st.sidebar.header("Enter Patient Information")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

hypertension = st.sidebar.selectbox(
    "Hypertension",
    ["No", "Yes"]
)

heart_disease = st.sidebar.selectbox(
    "Heart Disease",
    ["No", "Yes"]
)

smoking_history = st.sidebar.selectbox(
    "Smoking History",
    ["never", "former", "current", "not current"]
)

bmi = st.sidebar.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

hba1c = st.sidebar.number_input(
    "HbA1c Level",
    min_value=3.0,
    max_value=15.0,
    value=5.5
)

glucose = st.sidebar.number_input(
    "Blood Glucose Level",
    min_value=50,
    max_value=300,
    value=100
)




hypertension_val = 1 if hypertension == "Yes" else 0
heart_disease_val = 1 if heart_disease == "Yes" else 0

gender_encoded = le_gender.transform([gender])[0]
smoking_encoded = le_smoking.transform([smoking_history])[0]




input_data = {
    "gender": gender_encoded,
    "age": age,
    "hypertension": hypertension_val,
    "heart_disease": heart_disease_val,
    "smoking_history": smoking_encoded,
    "bmi": bmi,
    "HbA1c_level": hba1c,
    "blood_glucose_level": glucose
}




age_group = pd.cut(
    pd.Series([input_data["age"]]),
    bins=[0, 30, 50, 65, 120],
    labels=[0, 1, 2, 3]
).astype(int).iloc[0]

input_data["age_group"] = age_group


input_data["high_risk_comorbidity"] = int(
    (input_data["hypertension"] == 1) and
    (input_data["heart_disease"] == 1)
)


input_data["metabolic_risk"] = int(
    (input_data["HbA1c_level"] > 6.5) or
    (input_data["blood_glucose_level"] > 126)
)



input_df = pd.DataFrame([input_data])




original_input = input_df.copy()




input_df[["bmi", "HbA1c_level", "blood_glucose_level"]] = robust_scaler.transform(
    input_df[["bmi", "HbA1c_level", "blood_glucose_level"]]
)

input_df[["age"]] = std_scaler.transform(
    input_df[["age"]]
)



input_df = input_df[feature_columns]




input_df = input_df.astype(float)
input_df = input_df.reset_index(drop=True)




if st.button("Predict Diabetes Risk"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    st.metric("Diabetes Risk Probability", f"{probability:.2%}")

    st.progress(float(probability))

    if prediction == 1:
        st.error(" High Diabetes Risk")
    else:
        st.success(" Low Diabetes Risk")


   

    st.subheader("Explainable AI (XAI)")

    try:

        scaled_input = input_df.copy().astype(float)

        fig1, fig2, explanation_text = explain_prediction(
            model,
            scaled_input,
            original_input
        )

        st.write(explanation_text)

        st.subheader("Prediction Explanation (SHAP Waterfall)")
        st.pyplot(fig1)

        st.subheader("Feature Importance")
        st.pyplot(fig2)

    except Exception as e:

        st.warning(f"XAI explanation failed: {e}")




st.divider()
st.header("💬 Ask Health Questions")

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


user_input = st.chat_input(
    "Ask something about diabetes, health, etc..."
)


if user_input:

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    response = ask_llm(user_input)

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )




st.divider()

st.caption(
    "Note: This tool provides educational risk estimates and is not a medical diagnosis. "
    "Consult a healthcare professional for medical advice."
)