import streamlit as st
import pandas as pd
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.fairness_check.py import check_group_fairness # Need to ensure name is correct
from src.monitor import simulate_monitoring

st.set_page_config(page_title="Admin ML Monitoring", layout="wide")

st.title("🛰️ Model Evaluation & Monitoring Dashboard")

# Paths
project_root = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(project_root, 'data', 'loan_data.csv')
monitor_path = os.path.join(project_root, 'data', 'monitor_report.json')

# 1. Model Health
st.header("📈 Model Health & Performance")
col1, col2, col3 = st.columns(3)

# Load metrics from metadata
metadata_path = os.path.join(project_root, 'models', 'v1', 'metadata.json')
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
        metrics = meta['metrics']
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")
        col3.metric("Precision", f"{metrics['precision']:.2%}")

# 2. Responsible AI (Fairness)
st.header("⚖️ Responsible AI Check (Group Fairness)")
# Fixing import path internally
from src.fairness_check import check_group_fairness 
fair_report = check_group_fairness(data_path)

if isinstance(fair_report, dict):
    fcol1, fcol2 = st.columns(2)
    fcol1.write("### Demographic Parity Analysis")
    fcol1.json(fair_report)
    
    # Status indicator
    if fair_report['status'] == "PASS":
        fcol2.success("Fairness Status: PASS")
    else:
        fcol2.warning("Fairness Status: WARNING")

# 3. Drift Monitoring
st.header("👁️ Drift & Trend Monitoring")
if st.button("Run Live Monitoring Simulation"):
    simulate_monitoring()
    st.experimental_rerun()

if os.path.exists(monitor_path):
    with open(monitor_path, 'r') as f:
        mon = json.load(f)
        
    st.write("### Feature Drift Indicators")
    st.write(mon['feature_drifts'])
    
    st.write("### Approval Rate Trends (Last 30 Days)")
    st.line_chart(mon['daily_approval_rates'])
    
    if mon['active_alerts']:
        st.error(f"Active Alerts: {mon['active_alerts']}")
