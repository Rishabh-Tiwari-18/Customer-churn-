import streamlit as st
import pandas as pd
import joblib


# Load model

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")


st.set_page_config(page_title="Customer Churn Prediction")

st.title("🏦 Customer Churn Prediction System")


# Inputs

credit_score = st.number_input("Credit Score", 300, 900, 650)

country = st.selectbox(
    "Country",
    ["France", "Spain", "Germany"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.slider("Age", 18, 100, 35)

tenure = st.slider("Tenure", 0, 10, 5)

balance = st.number_input(
    "Balance",
    0.0,
    300000.0,
    50000.0
)

products_number = st.slider(
    "Products Number",
    1,
    4,
    1
)

credit_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    0.0,
    300000.0,
    50000.0
)


# Encoding

country_map = {
    "France": 0,
    "Spain": 1,
    "Germany": 2
}

gender_map = {
    "Male": 1,
    "Female": 0
}


# Dataframe

input_data = pd.DataFrame({
    "credit_score": [credit_score],
    "country": [country_map[country]],
    "gender": [gender_map[gender]],
    "age": [age],
    "tenure": [tenure],
    "balance": [balance],
    "products_number": [products_number],
    "credit_card": [credit_card],
    "active_member": [active_member],
    "estimated_salary": [estimated_salary]
})


# Scaling

scale_cols = [
    "credit_score",
    "age",
    "balance",
    "estimated_salary"
]

input_data[scale_cols] = scaler.transform(
    input_data[scale_cols]
)


# Prediction

if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("Customer likely to churn ⚠️")

    else:
        st.success("Customer likely to stay ✅")

    st.write(f"Churn Probability: {probability:.2f}")