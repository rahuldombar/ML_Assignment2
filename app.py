import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════
# CUSTOM CSS - Keep it clean and modern
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Main container */
    .main .block-container { padding-top: 1.5rem; }
    
    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .hero-banner h1 { margin: 0; font-size: 2.2rem; }
    .hero-banner p { margin: 0.5rem 0 0 0; opacity: 0.9; }
    
    /* Metric cards */
    .metric-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .metric-box h3 { margin: 0; color: #333; font-size: 0.9rem; }
    .metric-box p { margin: 0.3rem 0 0 0; font-size: 1.5rem; font-weight: bold; color: #667eea; }
    
    /* Model type badge */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-ensemble { background: #d4edda; color: #155724; }
    .badge-traditional { background: #d1ecf1; color: #0c5460; }
    
    /* Result box */
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .approved { background: #d4edda; border: 2px solid #28a745; }
    .rejected { background: #f8d7da; border: 2px solid #dc3545; }
    .result-box h2 { margin: 0; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MODEL INFO - Keep it informative but concise
# ════════════════════════════════════════════════════════════
MODEL_INFO = {
    "Logistic Regression": {
        "type": "Traditional",
        "icon": "📈",
        "desc": "Linear classifier - simple and interpretable. Good baseline model."
    },
    "Decision Tree": {
        "type": "Traditional",
        "icon": "🌳",
        "desc": "Tree-based model that learns decision rules. Easy to understand."
    },
    "KNN": {
        "type": "Traditional",
        "icon": "📍",
        "desc": "Classifies based on nearest neighbors. No training phase needed."
    },
    "Naive Bayes": {
        "type": "Traditional",
        "icon": "🎲",
        "desc": "Probabilistic classifier using Bayes theorem. Fast and efficient."
    },
    "Random Forest": {
        "type": "Ensemble",
        "icon": "🌲",
        "desc": "Ensemble of decision trees. Reduces overfitting through averaging."
    },
    "XGBoost": {
        "type": "Ensemble",
        "icon": "🚀",
        "desc": "Gradient boosting algorithm. Sequentially corrects prediction errors."
    }
}

# ════════════════════════════════════════════════════════════
# LOAD DATA & MODELS
# ════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    try:
        package = joblib.load("loan_models_complete.pkl")
        return package
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

@st.cache_data
def load_metrics():
    try:
        return pd.read_csv("model_comparison_metrics.csv")
    except:
        return None

# Load everything
model_package = load_models()
metrics_df = load_metrics()

if model_package is None:
    st.error("⚠️ Could not load models. Make sure model files are present.")
    st.stop()

models = model_package['models']
scaler = model_package['scaler']
feature_names = model_package['feature_names']

# ════════════════════════════════════════════════════════════
# SIDEBAR - Model Selection
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Model Selection")
    st.markdown("---")
    
    # Filter by type
    filter_type = st.radio("Filter by Type", ["All Models", "Ensemble Only", "Traditional Only"])
    
    if filter_type == "Ensemble Only":
        available_models = [m for m in MODEL_INFO if MODEL_INFO[m]["type"] == "Ensemble"]
    elif filter_type == "Traditional Only":
        available_models = [m for m in MODEL_INFO if MODEL_INFO[m]["type"] == "Traditional"]
    else:
        available_models = list(MODEL_INFO.keys())
    
    selected_model = st.selectbox("Choose Model", available_models)
    
    # Show model info
    info = MODEL_INFO[selected_model]
    badge_class = "badge-ensemble" if info["type"] == "Ensemble" else "badge-traditional"
    st.markdown(f'{info["icon"]} <span class="{badge_class}">{info["type"]}</span>', unsafe_allow_html=True)
    st.caption(info["desc"])
    
    st.markdown("---")
    
    # Show quick metrics
    if metrics_df is not None:
        model_row = metrics_df[metrics_df["ML Model Name"] == selected_model]
        if not model_row.empty:
            acc = model_row["Accuracy"].values[0]
            f1 = model_row["F1 Score"].values[0]
            st.metric("Accuracy", f"{acc:.2%}")
            st.metric("F1 Score", f"{f1:.2%}")

# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <h1>🏦 Loan Approval Prediction System</h1>
    <p>ML Assignment 2 · <strong>Rahul Dombar (2024dc04081)</strong></p>
    <p>Predict loan approval decisions using 6 different machine learning models</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MAIN CONTENT - Tabs
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔮 Predictions", "📊 Model Comparison", "ℹ️ About"])

# ─────────────────────────────────────────────────────────────
# TAB 1: HOME
# ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## 📌 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Problem
        Predict whether a loan application will be **Approved** or **Rejected** based on:
        - Applicant financial info (income, assets)
        - Credit score (CIBIL)
        - Employment status
        - Loan details
        
        ### Dataset
        - **Source:** Kaggle Loan Approval Dataset
        - **Samples:** 4,269 applications
        - **Features:** 13 (after preprocessing)
        - **Classes:** Approved (62%) / Rejected (38%)
        """)
    
    with col2:
        st.markdown("""
        ### Models Implemented
        **Traditional Models:**
        - Logistic Regression
        - Decision Tree
        - K-Nearest Neighbors
        - Naive Bayes
        
        **Ensemble Models:**
        - Random Forest
        - XGBoost
        
        ### Evaluation Metrics
        - Accuracy, AUC, Precision
        - Recall, F1 Score, MCC
        """)
    
    st.markdown("---")
    
    # Show feature importance for ensemble models
    if selected_model in ["Random Forest", "XGBoost", "Decision Tree"]:
        st.markdown(f"### 🔍 Feature Importance - {selected_model}")
        
        model = models[selected_model]
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(importance_df['Feature'], importance_df['Importance'], color='#667eea')
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Most Important Features')
            ax.invert_yaxis()
            st.pyplot(fig)

# ─────────────────────────────────────────────────────────────
# TAB 2: PREDICTIONS
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 🔮 Make Predictions")
    
    # File upload
    st.markdown("### 📤 Upload Test Data")
    st.info("Upload a CSV file with loan application data (same format as training data)")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            test_data = pd.read_csv(uploaded_file)
            
            st.success(f"✓ Loaded {len(test_data)} records")
            
            with st.expander("👁️ Preview data"):
                st.dataframe(test_data.head())
            
            # Check if target exists
            has_target = 'loan_status' in test_data.columns
            
            if has_target:
                y_true = test_data['loan_status']
                X_test = test_data.drop('loan_status', axis=1)
            else:
                X_test = test_data
                y_true = None
            
            # Align columns
            for col in feature_names:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[feature_names]
            
            # Make predictions
            if st.button("🚀 Run Prediction", type="primary"):
                model = models[selected_model]
                
                # Scale if needed
                if selected_model in ["Logistic Regression", "KNN", "Naive Bayes"]:
                    X_scaled = scaler.transform(X_test)
                    y_pred = model.predict(X_scaled)
                    y_prob = model.predict_proba(X_scaled)[:, 1]
                else:
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                
                st.success("✓ Predictions complete!")
                
                # Results
                st.markdown("### 📊 Results")
                
                results = pd.DataFrame({
                    'Prediction': ['Approved' if p == 1 else 'Rejected' for p in y_pred],
                    'Confidence': [f"{p*100:.1f}%" for p in y_prob]
                })
                
                if has_target:
                    results.insert(0, 'Actual', ['Approved' if t == 1 else 'Rejected' for t in y_true])
                    results['Match'] = ['✓' if p == t else '✗' for p, t in zip(y_pred, y_true)]
                
                st.dataframe(results.head(20), use_container_width=True)
                
                # Summary
                col1, col2 = st.columns(2)
                
                with col1:
                    approved = sum(y_pred == 1)
                    rejected = sum(y_pred == 0)
                    
                    st.markdown("**Prediction Summary**")
                    st.write(f"- Approved: {approved} ({approved/len(y_pred)*100:.1f}%)")
                    st.write(f"- Rejected: {rejected} ({rejected/len(y_pred)*100:.1f}%)")
                
                with col2:
                    if has_target:
                        acc = accuracy_score(y_true, y_pred)
                        st.markdown("**Performance**")
                        st.write(f"- Accuracy: {acc:.2%}")
                        st.write(f"- Correct: {sum(y_pred == y_true)}/{len(y_true)}")
                
                # Metrics and confusion matrix if ground truth available
                if has_target:
                    st.markdown("---")
                    st.markdown("### 📈 Evaluation Metrics")
                    
                    acc = accuracy_score(y_true, y_pred)
                    auc = roc_auc_score(y_true, y_prob)
                    prec = precision_score(y_true, y_pred)
                    rec = recall_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)
                    mcc = matthews_corrcoef(y_true, y_pred)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{acc:.4f}")
                    col1.metric("AUC", f"{auc:.4f}")
                    col2.metric("Precision", f"{prec:.4f}")
                    col2.metric("Recall", f"{rec:.4f}")
                    col3.metric("F1 Score", f"{f1:.4f}")
                    col3.metric("MCC", f"{mcc:.4f}")
                    
                    st.markdown("---")
                    st.markdown("### 🎯 Confusion Matrix")
                    
                    cm = confusion_matrix(y_true, y_pred)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Counts
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                              xticklabels=['Rejected', 'Approved'],
                              yticklabels=['Rejected', 'Approved'])
                    ax1.set_title('Confusion Matrix')
                    ax1.set_xlabel('Predicted')
                    ax1.set_ylabel('Actual')
                    
                    # Normalized
                    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
                              xticklabels=['Rejected', 'Approved'],
                              yticklabels=['Rejected', 'Approved'])
                    ax2.set_title('Normalized')
                    ax2.set_xlabel('Predicted')
                    ax2.set_ylabel('Actual')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Classification report
                    st.markdown("### 📋 Classification Report")
                    report = classification_report(y_true, y_pred, 
                                                 target_names=['Rejected', 'Approved'],
                                                 output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error: {e}")
    
    else:
        st.info("👆 Upload a CSV file to start making predictions")

# ─────────────────────────────────────────────────────────────
# TAB 3: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 📊 Model Performance Comparison")
    
    if metrics_df is not None:
        # Comparison table
        st.markdown("### 📋 All Models - Performance Metrics")
        
        styled = metrics_df.style.background_gradient(
            subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC'],
            cmap='RdYlGn'
        ).format({
            'Accuracy': '{:.4f}',
            'AUC': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1 Score': '{:.4f}',
            'MCC': '{:.4f}'
        })
        
        st.dataframe(styled, use_container_width=True)
        
        # Best model
        best = metrics_df.loc[metrics_df['F1 Score'].idxmax()]
        st.success(f"🏆 Best Model: **{best['ML Model Name']}** (F1: {best['F1 Score']:.4f})")
        
        st.markdown("---")
        
        # Visual comparison
        st.markdown("### 📈 Visual Comparison")
        
        metrics_to_plot = st.multiselect(
            "Select metrics to compare",
            ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC'],
            default=['Accuracy', 'F1 Score', 'AUC']
        )
        
        if metrics_to_plot:
            fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6*len(metrics_to_plot), 5))
            if len(metrics_to_plot) == 1:
                axes = [axes]
            
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
            
            for ax, metric in zip(axes, metrics_to_plot):
                data = metrics_df.sort_values(metric, ascending=True)
                ax.barh(data['ML Model Name'], data[metric], color=colors)
                ax.set_xlabel(metric)
                ax.set_xlim(0, 1)
                ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

# ─────────────────────────────────────────────────────────────
# TAB 4: ABOUT
# ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### Assignment Details
    - **Course:** Machine Learning
    - **Program:** M.Tech Data science and Engineering
    - **Institution:** BITS Pilani WILP
    - **Assignment:** 2
    - **Student:** Rahul Dombar (2024dc04081)
    
    ### Objective
    Build and compare 6 classification models for predicting loan approval decisions.
    
    ### Technologies Used
    - Python 3.8+
    - scikit-learn (ML algorithms)
    - XGBoost (gradient boosting)
    - Streamlit (web interface)
    - Pandas, NumPy (data processing)
    - Matplotlib, Seaborn (visualization)
    
    ### Models
    1. **Logistic Regression** - Linear baseline
    2. **Decision Tree** - Tree-based classifier
    3. **KNN** - Instance-based learning
    4. **Naive Bayes** - Probabilistic classifier
    5. **Random Forest** - Ensemble method (bagging)
    6. **XGBoost** - Ensemble method (boosting)
    
    ### Dataset
    - Source: Kaggle Loan Approval Dataset
    - Size: 4,269 samples
    - Features: 13 (after encoding)
    - Target: Binary (Approved/Rejected)
    
    ### Repository
    All code and models are available on GitHub.
    
    ---
    

    """)

# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>ML Assignment 2 · Rahul Dombar (2024dc04081) · BITS Pilani</p>
</div>
""", unsafe_allow_html=True)
