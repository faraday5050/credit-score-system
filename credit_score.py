# app.py - Complete Loan Credit Risk Predictor with BVN Integration
# Your original code is PRESERVED - BVN features ADDED without breaking anything

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import random  # Added for BVN simulation

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Loan Credit Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (YOUR ORIGINAL + minor additions for BVN)
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
    /* NEW: BVN section styling - won't affect your existing styles */
    .bvn-section {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1E3A8A;
        margin-bottom: 20px;
    }
    .credit-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: bold;
    }
    .badge-good { background-color: #10B981; color: white; }
    .badge-fair { background-color: #F59E0B; color: white; }
    .badge-poor { background-color: #EF4444; color: white; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD SAVED MODELS AND ARTIFACTS (YOUR ORIGINAL CODE - UNTOUCHED)
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
# NEW: BVN SIMULATOR CLASS (ADDED - DOESN'T AFFECT YOUR EXISTING CODE)
# =============================================================================

class BVNSimulator:
    """
    Simulates Bank Verification Number (BVN) integration
    This runs alongside your existing code without modifying it
    """
    
    def __init__(self):
        self.bvn_database = self._create_sample_profiles()
    
    def _create_sample_profiles(self):
        """Create realistic BVN profiles"""
        return {
            # Good credit profile
            '12345678901': {
                '': 720,
                'credit_score': 500,
                'score_band': 'GOOD',
                'existing_loans': 1,
                'total_outstanding': 250000,
                'monthly_obligations': 12500,
                'late_payments_30d': 1,
                'late_payments_60d': 0,
                'late_payments_90d': 0,
                'defaults': 0,
                'credit_age_years': 7,
                'name': 'John Adebayo',
                'bank': 'Access Bank',
                'phone_verified': True,
                'email_verified': True
            },
            # Fair credit profile
            '23456789012': {
                'credit_score': 650,
                'score_band': 'FAIR',
                'existing_loans': 2,
                'total_outstanding': 450000,
                'monthly_obligations': 22500,
                'late_payments_30d': 2,
                'late_payments_60d': 1,
                'late_payments_90d': 0,
                'defaults': 0,
                'credit_age_years': 4,
                'name': 'Blessing Okafor',
                'bank': 'First Bank',
                'phone_verified': True,
                'email_verified': False
            },
            # Poor credit profile
            '34567890123': {
                'credit_score': 520,
                'score_band': 'POOR',
                'existing_loans': 3,
                'total_outstanding': 680000,
                'monthly_obligations': 34000,
                'late_payments_30d': 4,
                'late_payments_60d': 2,
                'late_payments_90d': 1,
                'defaults': 1,
                'credit_age_years': 2,
                'name': 'Chuka Eze',
                'bank': 'GTBank',
                'phone_verified': True,
                'email_verified': False
            },
            # Thin file profile
            '45678901234': {
                'credit_score': 580,
                'score_band': 'THIN FILE',
                'existing_loans': 1,
                'total_outstanding': 150000,
                'monthly_obligations': 12500,
                'late_payments_30d': 0,
                'late_payments_60d': 0,
                'late_payments_90d': 0,
                'defaults': 0,
                'credit_age_years': 1,
                'name': 'Fatima Abubakar',
                'bank': 'UBA',
                'phone_verified': True,
                'email_verified': False
            }
        }
    
    def lookup_bvn(self, bvn_number):
        """
        Simulate BVN lookup - returns credit data
        For unknown BVNs, generates realistic random data
        """
        bvn_number = str(bvn_number).replace(' ', '')
        
        # Validate format
        if not bvn_number.isdigit() or len(bvn_number) != 11:
            return {
                'success': False,
                'error': 'Invalid BVN format. Must be 11 digits.'
            }
        
        # Check if in predefined database
        if bvn_number in self.bvn_database:
            data = self.bvn_database[bvn_number].copy()
            data['success'] = True
            data['message'] = 'BVN found in credit bureau'
            return data
        
        # Generate random profile for unknown BVNs
        random.seed(int(bvn_number[-6:]))  # Deterministic based on BVN
        
        credit_score = random.randint(500, 780)
        if credit_score >= 700:
            band = 'GOOD'
            defaults = 0
            late_payments = random.randint(0, 1)
        elif credit_score >= 600:
            band = 'FAIR'
            defaults = 0
            late_payments = random.randint(1, 3)
        else:
            band = 'POOR'
            defaults = random.randint(0, 1)
            late_payments = random.randint(2, 5)
        
        return {
            'success': True,
            'message': 'BVN lookup successful (simulated)',
            'credit_score': credit_score,
            'score_band': band,
            'existing_loans': random.randint(0, 4),
            'total_outstanding': random.randint(0, 800000),
            'monthly_obligations': random.randint(0, 40000),
            'late_payments_30d': late_payments,
            'late_payments_60d': max(0, late_payments - random.randint(0, 2)),
            'late_payments_90d': max(0, late_payments - random.randint(1, 3)),
            'defaults': defaults,
            'credit_age_years': random.randint(0, 10),
            'bank': random.choice(['Access', 'First', 'GTB', 'UBA', 'Zenith']),
            'phone_verified': random.random() > 0.2
        }
    
    def get_credit_badge(self, score):
        """Return HTML badge based on credit score"""
        if score >= 700:
            return '<span class="credit-badge badge-good">🟢 GOOD</span>'
        elif score >= 600:
            return '<span class="credit-badge badge-fair">🟡 FAIR</span>'
        else:
            return '<span class="credit-badge badge-poor">🔴 POOR</span>'

# Initialize BVN simulator
bvn_simulator = BVNSimulator()

# =============================================================================
# YOUR ACTUAL COLUMN NAMES (YOUR ORIGINAL - UNTOUCHED)
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
# HELPER FUNCTIONS (YOUR ORIGINAL - UNTOUCHED)
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
    
    # Reorder columns to match training EXACTLY
    df = df[feature_columns]
    
    # Apply scaling - BUT only to columns the scaler expects
    if artifacts and artifacts['scaler']:
        # Get the exact columns the scaler was trained on
        scaler_columns = artifacts['scaler'].feature_names_in_
        
        # Only scale columns that exist in the scaler
        cols_to_scale = [col for col in scaler_columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
        
        # Scale in the correct order
        if cols_to_scale:
            df[cols_to_scale] = artifacts['scaler'].transform(df[cols_to_scale])
    
    return df
# =============================================================================
# MAIN APP INTERFACE (YOUR ORIGINAL - WITH BVN ADDED IN SIDEBAR ONLY)
# =============================================================================

st.markdown('<p class="main-header">🏦 Credit Risk Assessment Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">For Financial Inclusion Lending | Powered by Machine Learning</p>', unsafe_allow_html=True)

# Sidebar - YOUR ORIGINAL CONTENT + BVN SECTION ADDED AT THE BOTTOM
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
    
    # =========================================================================
    # NEW: BVN INTEGRATION SECTION (ADDED - DOESN'T AFFECT YOUR MAIN FUNCTIONALITY)
    # =========================================================================
    st.markdown("---")
    with st.expander("🆔 BVN Credit Lookup (Optional)", expanded=False):
        st.markdown("""
        **Simulated BVN Integration**  
        In production, this connects to NIBSS Credit Bureau
        """)
        
        bvn_input = st.text_input(
            "Enter 11-digit BVN",
            max_chars=11,
            placeholder="e.g., 12345678901",
            key="bvn_input"
        )
        
        if bvn_input:
            bvn_result = bvn_simulator.lookup_bvn(bvn_input)
            
            if bvn_result['success']:
                # Display credit score with badge
                score = bvn_result['credit_score']
                badge_html = bvn_simulator.get_credit_badge(score)
                st.markdown(f"**Credit Score:** {score} {badge_html}", unsafe_allow_html=True)
               
                
                # Create columns for metrics
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    st.metric("Credit Age", f"{bvn_result['credit_age_years']} yrs")
                    st.metric("Existing Loans", bvn_result['existing_loans'])
                with bcol2:
                    st.metric("Late Payments", bvn_result['late_payments_30d'])
                    st.metric("Defaults", bvn_result['defaults'])
                
                # Store in session state for later use
                st.session_state['bvn_data'] = bvn_result
                st.session_state['bvn_enriched'] = True
                
                # Show verification status
                if bvn_result.get('phone_verified'):
                    st.success("📱 Phone Verified")
                
                st.caption(f"Source: {bvn_result.get('bank', 'Credit Bureau')}")
            else:
                st.error(bvn_result['error'])
                st.session_state['bvn_enriched'] = False

# Main content - YOUR ORIGINAL TABS (COMPLETELY UNTOUCHED)
tab1, tab2, tab3 = st.tabs(["📝 Prediction", "📊 Model Performance", "ℹ️ About"])

# =============================================================================
# TAB 1: PREDICTION (YOUR ORIGINAL CODE - WITH BVN ENHANCEMENTS ADDED CAREFULLY)
# =============================================================================
with tab1:
    st.markdown("### Enter Applicant Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Personal Information")
        
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=5)
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=50000, value=5000, step=100)
        employment_length = st.slider("Employment Length (years)", 0.0, 50.0, 5.0, step=0.5)
        home_ownership = st.selectbox(
            "Home Ownership",
            options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
            index=0
        )
        
        st.markdown("#### 📱 Credit History")
        credit_history_length = st.slider("Credit History Length (years)", 0, 20, 5)
        historical_default = st.radio(
            "Previous Default History",
            options=['No', 'Yes'],
            index=0,
            horizontal=False
        )
        historical_default = 'NO' if historical_default == 'No' else 'YES'
        
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

        age_options = ['<20', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-70']

        st.selectbox('age', 
                        options = age_options)
    
    with col2:
        st.markdown("#### 💰 Loan Details")
        
        loan_amount = st.number_input("Loan Amount Requested ($)", min_value=500, max_value=50000, value=15000, step=500)
        loan_intent = st.selectbox(
            "Loan Purpose",
            options=['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOME IMPROVEMENT', 'DEBT CONSOLIDATION'],
            index=0
        )
        loan_grade = st.select_slider(
            "Loan Grade (A = Best, G = Worst)",
            options=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            value='B'
        )
        interest_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.5, step=0.1)
        
        # Calculate derived feature
        loan_percent_income = round((loan_amount / monthly_income) if monthly_income > 0 else 0, 3)
    
    # =========================================================================
    # NEW: BVN ENRICHMENT BANNER (SHOWS IF BVN WAS LOOKED UP)
    # =========================================================================
    if st.session_state.get('bvn_enriched', False):
        bvn_data = st.session_state.get('bvn_data', {})
        st.markdown(f"""
        <div class="bvn-section">
            <b>🆔 BVN Enriched Data</b> | Credit Score: {bvn_data.get('credit_score', 'N/A')} 
            ({bvn_data.get('score_band', 'N/A')}) | 
            Existing Obligations: ₦{bvn_data.get('monthly_obligations', 0):,}/month
        </div>
        """, unsafe_allow_html=True)
    
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
            
            # =================================================================
            # OPTIONAL: Enhance with BVN data if available (DOESN'T BREAK WITHOUT IT)
            # =================================================================
            if st.session_state.get('bvn_enriched', False):
                bvn_data = st.session_state.get('bvn_data', {})
                # Only add if columns exist in your model
                if 'Credit_Score' in artifacts['features']:
                    input_data['Credit_Score'] = bvn_data.get('credit_score', 0)
                if 'BVN_Verified' in artifacts['features']:
                    input_data['BVN_Verified'] = 1 if bvn_data.get('phone_verified', False) else 0
            
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
            
            # =================================================================
            # NEW: Enhanced Risk Factors with BVN Data
            # =================================================================
            if st.session_state.get('bvn_enriched', False):
                with st.expander("📋 Enhanced Risk Factors (Including BVN Data)"):
                    bvn_data = st.session_state.get('bvn_data', {})
                    ef1, ef2, ef3 = st.columns(3)
                    with ef1:
                        st.metric("BVN Credit Score", bvn_data.get('credit_score', 'N/A'))
                        st.metric("Existing Loans", bvn_data.get('existing_loans', 0))
                    with ef2:
                        st.metric("Monthly Obligations", f"₦{bvn_data.get('monthly_obligations', 0):,}")
                        dti_enhanced = round((bvn_data.get('monthly_obligations', 0) / monthly_income) * 100, 1) if monthly_income > 0 else 0
                        st.metric("Enhanced DTI", f"{dti_enhanced}%")
                    with ef3:
                        st.metric("Late Payments", bvn_data.get('late_payments_30d', 0))
                        st.metric("Credit Age", f"{bvn_data.get('credit_age_years', 0)} yrs")
            
            # Probability chart (YOUR ORIGINAL)
            fig = px.bar(
                x=['Repay', 'Default'],
                y=probabilities,
                color=['Repay', 'Default'],
                color_discrete_map={'Repay': '#10B981', 'Default': '#EF4444'},
                title="Prediction Probabilities"
            )
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: MODEL PERFORMANCE (YOUR ORIGINAL - UNTOUCHED)
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
# TAB 3: ABOUT (YOUR ORIGINAL - WITH BVN MENTION ADDED)
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
    
    ### 🆔 BVN Integration (Demo)
    This system simulates integration with Nigeria's **Bank Verification Number** 
    system, demonstrating how real fintechs access:
    - Credit bureau data
    - Verified identity (KYC)
    - Existing loan obligations across banks
    - Payment behavior history
    
    ### 🔬 Technical Approach
    - **Data:** 32,566 loan records
    - **Model:** Random Forest Classifier
    - **Performance:** 90.5% accuracy, 81.9% precision, 72.8% recall

    ### 🔮 Developer
    - **NAME:** Yahaya Eneojo Michael
    - **COHORT:** NextGen Cohort
    - **LOCATION:** Gidan Kwanu Minna Niger State
    
    ### 📝 AI Disclosure
    This project was developed with assistance from AI tools for code structure 
    and optimization. All core ML implementation was performed by the developer.
    """)

# Footer (YOUR ORIGINAL)
st.markdown("---")
st.markdown("© 2026 Credit Risk Predictor | NextGen Knowledge Showcase | Financial Inclusion Pillar")
