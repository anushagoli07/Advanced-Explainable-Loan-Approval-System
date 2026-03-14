import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Page Config
st.set_page_config(page_title="Loan Approval AI", layout="wide")

st.title("🛡️ Explainable Loan Approval System")
st.markdown("### Responsible AI: Predict and Explain Loan Decisions")

# Sidebar - User Input
st.sidebar.header("Application Details")
income = st.sidebar.number_input("Annual Income ($)", 20000, 200000, 50000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
debt_amount = st.sidebar.number_input("Total Debt ($)", 0, 100000, 15000)
employment_years = st.sidebar.slider("Employment Years", 0, 40, 5)
age = st.sidebar.slider("Age", 18, 75, 30)

# API endpoint
API_URL = "http://127.0.0.1:8000/predict"

if st.sidebar.button("Submit Application"):
    payload = {
        "income": income,
        "credit_score": credit_score,
        "debt_amount": debt_amount,
        "employment_years": employment_years,
        "age": age
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            
            # Decision Card
            decision = result["decision"]
            color = "green" if decision == "APPROVED" else "red"
            st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; border: 2px solid {color}; background-color: rgba(0,0,0,0.05)'>
                <h2 style='color: {color}; margin-top: 0;'>Decision: {decision}</h2>
                <p>Confidence: {result['approval_probability']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Explanation Section
            st.header("🔍 Why this decision?")
            st.write("SHAP (Shapley Additive Explanations) identifies which features contributed most to the model's decision.")
            
            # Create a dataframe for the waterfall visualization
            df_exp = pd.DataFrame(result["explanations"])
            
            # Simple bar chart for impact
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['green' if x > 0 else 'red' for x in df_exp['impact_score']]
            ax.barh(df_exp['feature'], df_exp['impact_score'], color=colors)
            ax.set_xlabel("Impact on Approval (SHAP Value)")
            ax.set_title("Feature Contribution to Prediction")
            st.pyplot(fig)
            
            st.table(df_exp[['feature', 'direction', 'impact_score']])
            
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Could not connect to API: {e}")

st.sidebar.markdown("---")
st.sidebar.info("This system uses SHAP to provide transparent credit scoring decisions.")
