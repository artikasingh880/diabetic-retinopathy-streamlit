import streamlit as st 
import numpy as np 
import joblib 
import pandas as pd 
import matplotlib.pyplot as plt 
 
try: 
    model = joblib.load("logistic_model.pkl") 
    scaler = joblib.load("scaler.pkl") 
except FileNotFoundError: 
    st.error("Model or scaler file not found. Ensure 'logistic_model.pkl' and 'scaler.pkl' are present.") 
    st.stop() 
 
st.set_page_config(page_title="Diabetic Retinopathy Prediction", layout="centered") 
st.title("Diabetic Retinopathy Prediction") 
st.markdown("Enter patient details to predict the likelihood of diabetic retinopathy using a logistic regression model.") 
 
st.subheader("Patient Information") 
age = st.number_input("Age (years)", min_value=35, max_value=103, value=50, step=1, help="Patient's age (range: 35-103)") 
systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=69, max_value=151, value=120, step=1, help="Systolic BP (range: 69-151)") 
diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=62, max_value=133, value=80, step=1, help="Diastolic BP (range: 62-133)") 
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=69, max_value=148, value=100, step=1, help="Cholesterol level (range: 69-148)") 
 
input_data = np.array([[age, systolic_bp, diastolic_bp, cholesterol]]) 
 
if st.button("Predict"): 
    try: 
        input_data_scaled = scaler.transform(input_data) 
        prediction = model.predict(input_data_scaled)[0] 
        if prediction == 1: 
            st.error("**Prediction: Diabetic Retinopathy Detected**") 
            st.markdown("Please consult a healthcare professional for further evaluation.") 
        else: 
            st.success("**Prediction: No Diabetic Retinopathy Detected**") 
            st.markdown("Continue monitoring health metrics regularly.") 
    except Exception as e: 
        st.error(f"An error occurred during prediction: {str(e)}") 
 
st.markdown("---") 
st.subheader("Dependency Weightage") 
st.markdown("The following table and graph show the importance of each Python package used in this project.") 
 
dependencies = { 
    "Package": ["streamlit", "scikit-learn", "numpy", "joblib", "pandas"], 
    "Weightage (%)": [40, 25, 20, 10, 5], 
    "Purpose": [ 
        "Web app interface", 
        "Model loading and prediction", 
        "Numerical operations, input arrays", 
        "Model file loading", 
        "Optional data handling" 
    ] 
} 
dep_df = pd.DataFrame(dependencies) 
st.table(dep_df) 
 
fig, ax = plt.subplots(figsize=(8, 6)) 
ax.bar(dep_df["Package"], dep_df["Weightage (%)"], color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB3E6']) 
ax.set_xlabel("Packages") 
ax.set_ylabel("Weightage (%)") 
ax.set_title("Dependency Weightage for Diabetic Retinopathy App") 
ax.set_ylim(0, 50) 
for i, v in enumerate(dep_df["Weightage (%)"]): 
    ax.text(i, v + 1, f"{v}%%", ha='center', fontweight='bold') 
plt.tight_layout() 
st.pyplot(fig) 
 
st.markdown("---") 
st.subheader("About the Model") 
st.markdown("- **Model**: Logistic Regression\n- **Accuracy**: 77%%\n- **AUC Score**: 0.768\n- **Features Used**: Age, Systolic BP, Diastolic BP, Cholesterol\n- **Data Source**: Synthetic dataset with 6000 patient records") 
