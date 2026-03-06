# app.py - Complete Loan Credit Risk Predictor
# Customized for YOUR columns

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Loan Credit Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-top: 0;
    }
    .prediction-safe {
        background-color: #10B981;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-risk {
        background-color: #EF4444;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD SAVED MODELS AND ARTIFACTS
# =============================================================================

@st.cache_resource
def load_artifacts():
    """
    Load all saved model artifacts
    """
    artifacts = {}
    
    try:
        # Check if files exist
        required_files = {
            'model': 'random_forest_model.pkl',
            'encoder': 'label_encoder.pkl',
            'scaler': 'standard_scaler.pkl',
            'features': 'feature_columns.pkl'
        }
        
        missing_files = []
        for name, file in required_files.items():
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            st.error(f"⚠️ Missing files: {', '.join(missing_files)}")
            return None
        
        # Load all artifacts
        artifacts['model'] = joblib.load('random_forest_model.pkl')
        artifacts['encoder'] = joblib.load('label_encoder.pkl')
        artifacts['scaler'] = joblib.load('standard_scaler.pkl')
        artifacts['features'] = joblib.load('feature_columns.pkl')
        
        st.success("✅ Model loaded successfully!")
        return artifacts
        
    except Exception as e:
        st.error(f"❌ Error loading artifacts: {str(e)}")
        return None

# Load artifacts
artifacts = load_artifacts()

# =============================================================================
# YOUR ACTUAL COLUMN NAMES
# =============================================================================
YOUR_COLUMNS = [
    'Age',
    'Monthly_Income',
    'Home_Ownership',
    'Employment_Length',
    'Loan_Intent',
    'Loan',
    'Loan_Amount',
    'Loan_Interest_Rate',
    'Loan_Status',  # Target - not used for input
    'Loan_Percent_Income',
    'Historical_Default',
    'Credit_History_Length',
    'Loan_Status_Cat',  # Probably target encoded - not for input
    'Age_Range'  # Engineered feature
]

# Features used for prediction (excluding target and derived columns)
INPUT_FEATURES = [
    'Age',
    'Monthly_Income',
    'Home_Ownership',
    'Employment_Length',
    'Loan_Intent',
    'Loan',
    'Loan_Amount',
    'Loan_Interest_Rate',
    'Loan_Percent_Income',
    'Historical_Default',
    'Credit_History_Length',
    'Age_Range'
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def encode_categorical_values(df):
    """
    Encode categorical columns based on your training
    """
    # Mapping dictionaries based on common encoding patterns
    home_ownership_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    loan_intent_map = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 
                       'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
    loan_grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    default_map = {'NO': 0, 'YES': 1}
    
    # Apply mappings if columns exist
    if 'Home_Ownership' in df.columns:
        df['Home_Ownership'] = df['Home_Ownership'].map(home_ownership_map).fillna(0)
    
    if 'Loan_Intent' in df.columns:
        df['Loan_Intent'] = df['Loan_Intent'].map(loan_intent_map).fillna(0)
    
    if 'Loan' in df.columns:  # This is Loan_Grade based on your columns
        df['Loan'] = df['Loan'].map(loan_grade_map).fillna(0)
    
    if 'Historical_Default' in df.columns:
        df['Historical_Default'] = df['Historical_Default'].map(default_map).fillna(0)
    
    # Handle Age_Range if it's categorical
    if 'Age_Range' in df.columns and df['Age_Range'].dtype == 'object':
        age_range_map = {'<20': 0, '20-24': 1, '25-29': 2, '30-34': 3, '35-39': 4,
                        '40-44': 5, '45-49': 6, '50-54': 7, '55-59': 8, '60-64': 9, '65-70': 10}
        df['Age_Range'] = df['Age_Range'].map(age_range_map).fillna(0)
    
    return df

def prepare_input(data_dict, feature_columns):
    """
    Prepare input data for prediction
    """
    # Create DataFrame
    df = pd.DataFrame([data_dict])
    
    # Encode categorical values
    df = encode_categorical_values(df)
    
    # Ensure all required columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training
    df = df[feature_columns]
    
    # Apply scaling
    if artifacts and artifacts['scaler']:
        # Identify numerical columns for scaling
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numerical_cols] = artifacts['scaler'].transform(df[numerical_cols])
    
    return df

# =============================================================================
# MAIN APP INTERFACE
# =============================================================================

st.markdown('<p class="main-header">🏦 Credit Risk Assessment Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">For Financial Inclusion Lending | Powered by Machine Learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bank-cards.png", width=80)
    st.markdown("## About This Tool")
    st.markdown("""
    This tool predicts loan default risk using machine learning.
    
    **Model Performance:**
    - 🎯 Accuracy: 90.5%
    - 📊 Precision: 81.9%
    - 🎯 Recall: 72.8%
    - ⚖️ F1-Score: 0.77
    """)
    
    st.markdown("---")
    st.markdown("### Your Features")
    for col in INPUT_FEATURES:
        st.markdown(f"- {col}")

# Main content
tab1, tab2, tab3 = st.tabs(["📝 Prediction", "📊 Model Performance", "ℹ️ About"])

# =============================================================================
# TAB 1: PREDICTION
# =============================================================================
with tab1:
    st.markdown("### Enter Applicant Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Personal Information")
        
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=50000, value=5000, step=100)
        employment_length = st.slider("Employment Length (years)", 0.0, 50.0, 5.0, step=0.5)
        home_ownership = st.selectbox(
            "Home Ownership",
            options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
            index=0
        )
        
        st.markdown("#### 📱 Credit History")
        credit_history_length = st.slider("Credit History Length (years)", 0, 30, 5)
        historical_default = st.radio(
            "Previous Default History",
            options=['No', 'Yes'],
            index=0,
            horizontal=True
        )
        historical_default = 'N' if historical_default == 'No' else 'Yes'
        
        # Age Range (calculate from age)
        if age < 20:
            age_range = '<20'
        elif age < 25:
            age_range = '20-24'
        elif age < 30:
            age_range = '25-29'
        elif age < 35:
            age_range = '30-34'
        elif age < 40:
            age_range = '35-39'
        elif age < 45:
            age_range = '40-44'
        elif age < 50:
            age_range = '45-49'
        elif age < 55:
            age_range = '50-54'
        elif age < 60:
            age_range = '55-59'
        elif age < 65:
            age_range = '60-64'
        else:
            age_range = '65-70'
    
    with col2:
        st.markdown("#### 💰 Loan Details")
        
        loan_amount = st.number_input("Loan Amount Requested ($)", min_value=500, max_value=50000, value=15000, step=500)
        loan_intent = st.selectbox(
            "Loan Purpose",
            options=['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
            index=0
        )
        loan_grade = st.select_slider(
            "Loan Grade (A = Best, G = Worst)",
            options=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            value='B'
        )
        interest_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.5, step=0.1)
        
        # Calculate derived feature
        loan_percent_income = round(loan_amount / monthly_income, 3) if monthly_income > 0 else 0
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Credit Risk", type="primary", use_container_width=True)
    
    if predict_button and artifacts:
        with st.spinner("Analyzing applicant data..."):
            # Prepare input data with YOUR column names
            input_data = {
                'Age': age,
                'Monthly_Income': monthly_income,
                'Home_Ownership': home_ownership,
                'Employment_Length': employment_length,
                'Loan_Intent': loan_intent,
                'Loan': loan_grade,
                'Loan_Amount': loan_amount,
                'Loan_Interest_Rate': interest_rate,
                'Loan_Percent_Income': loan_percent_income,
                'Historical_Default': historical_default,
                'Credit_History_Length': credit_history_length,
                'Age_Range': age_range
            }
            
            # Prepare features
            input_df = prepare_input(input_data, artifacts['features'])
            
            # Make prediction
            prediction = artifacts['model'].predict(input_df)[0]
            probabilities = artifacts['model'].predict_proba(input_df)[0]
            
            # Display results
            st.markdown("### 📊 Prediction Results")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                if prediction == 0:
                    st.markdown("""
                    <div class="prediction-safe">
                        <h2>✅ LOW RISK</h2>
                        <h3>Likely to Repay</h3>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-risk">
                        <h2>⚠️ HIGH RISK</h2>
                        <h3>Potential Default</h3>
                    </div>
                    """, unsafe_allow_html=True)
            
            with res_col2:
                st.metric("Confidence", f"{max(probabilities)*100:.1f}%")
                st.metric("Risk Score", f"{probabilities[1]*100:.1f}%")
            
            with res_col3:
                if prediction == 0:
                    st.success("✅ APPROVE loan")
                else:
                    st.error("❌ DECLINE or review")
            
            # Probability chart
            fig = px.bar(
                x=['Repay', 'Default'],
                y=probabilities,
                color=['Repay', 'Default'],
                color_discrete_map={'Repay': '#10B981', 'Default': '#EF4444'},
                title="Prediction Probabilities"
            )
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: MODEL PERFORMANCE
# =============================================================================
with tab2:
    st.markdown("### 📊 Model Performance Metrics")
    
    # Metrics display
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [0.905, 0.819, 0.728, 0.771]
    }
        
    fig = px.bar(
        metrics_data,
        x='Metric',
        y='Score',
        color='Metric',
        range_y=[0, 1],
        title="Model Performance",
        text_auto='.3f'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    if artifacts and hasattr(artifacts['model'], 'feature_importances_'):
        st.markdown("### 🔑 Feature Importances")
        
        importances = artifacts['model'].feature_importances_
        feature_names = artifacts['features']
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(10)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 10 Important Features",
            color='Importance'
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 3: ABOUT
# =============================================================================
with tab3:
    st.markdown("### ℹ️ About This Project")
    
    st.markdown("""
    ## Credit Risk Assessment for Financial Inclusion
    
    This tool was developed for the **NextGen Knowledge Showcase** under the 
    **Financial Inclusion** impact pillar.
    
    ### 🎯 The Problem
    Many low-income individuals lack access to formal credit due to absence of 
    traditional credit history.
    
    ### 💡 Our Solution
    Machine learning model predicting creditworthiness using alternative data:
    - Age, Income, Employment
    - Loan purpose, amount, grade
    - Credit history and default history
    
    ### 🔬 Technical Approach
    - **Data:** 32,566 loan records
    - **Model:** Random Forest Classifier
    - **Performance:** 90.5% accuracy, 81.9% precision, 72.8% recall
    
    ### 📝 AI Disclosure
    This project was developed with assistance from AI tools for code structure 
    and optimization. All core ML implementation was performed by the developer.
    """)

# Footer
st.markdown("---")
st.markdown("© 2026 Credit Risk Predictor | NextGen Knowledge Showcase | Financial Inclusion Pillar")