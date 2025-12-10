# 1 Good - Lower Risk
# 0 Bad - Higher Risk
import streamlit as st
import pandas as pd
import joblib


model = joblib.load('random_forest_model.pkl')

encoders = {
    'Sex': joblib.load('Sex_encoder.pkl'),
    'Housing': joblib.load('Housing_encoder.pkl'),
    'Saving_accounts': joblib.load('Saving_accounts_encoder.pkl'),
    'Checking_account': joblib.load('Checking_account_encoder.pkl')
}

st.title("Credit Risk Prediction App")
st.write("Enter the applicant details below to predict credit risk.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.text_input("Job (0 - 3)", value="1") 
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)


input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [int(job)],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving_accounts"].transform([saving_accounts])[0]], 
    "Checking account": [encoders["Checking_account"].transform([checking_account])[0]], 
    "Credit amount_log": [np.log1p(credit_amount)],
    "Duration": [duration]
})

if st.button("Predict Credit Risk"):
    prediction = model.predict(input_df)[0]
    

    risk = "Good (Lower Risk)" if prediction == 1 else "Bad (Higher Risk)"
    

    if prediction == 1:
        st.success(f"The predicted credit risk is: **{risk}**")
    else:
        st.error(f"The predicted credit risk is: **{risk}**")