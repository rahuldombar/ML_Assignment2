# 🏦 Loan Approval Prediction - ML Assignment 2

**Rahul Dombar (2024dc04081)**  
M.Tech (DSE) - Machine Learning  
BITS Pilani WILP

---

## Problem Statement

Predict whether a loan application will be **Approved** or **Rejected** based on the applicant's financial information, credit history, employment status, and asset valuations. 

This is a binary classification problem where the goal is to build machine learning models that can automate loan approval decisions by analyzing patterns in historical loan application data. The system helps financial institutions make faster, more consistent lending decisions while identifying key factors that influence loan approval outcomes.

---

## Dataset Description

### Source
- **Dataset Name**: Loan Approval Dataset
- **Source**: Kaggle - Loan Approval Classification Dataset 
- **Link**:  https://www.kaggle.com/datasets/vishavgupta01/loan-approval

### Dataset Characteristics
- **Total Samples**: 4,269 loan applications
- **Total Features**: 13 (after preprocessing)
- **Problem Type**: Binary Classification
- **Target Variable**: loan_status (Approved / Rejected)
- **Class Distribution**:
  - Approved: 2,656 applications (62.2%)
  - Rejected: 1,613 applications (37.8%)

### Feature Description

| Feature Name | Type | Description |
|--------------|------|-------------|
| no_of_dependents | Numerical | Number of dependents of the loan applicant |
| education | Categorical | Education level (Graduate / Not Graduate) |
| self_employed | Categorical | Self-employment status (Yes / No) |
| income_annum | Numerical | Annual income of the applicant in currency units |
| loan_amount | Numerical | Total loan amount requested by the applicant |
| loan_term | Numerical | Loan repayment period in years |
| cibil_score | Numerical | Credit score ranging from 300 to 900 |
| residential_assets_value | Numerical | Total value of residential properties owned |
| commercial_assets_value | Numerical | Total value of commercial properties owned |
| luxury_assets_value | Numerical | Value of luxury assets (vehicles, jewelry, etc.) |
| bank_asset_value | Numerical | Total bank account balance and deposits |

### Data Quality
- **Missing Values**: None (0 missing values across all features)
- **Duplicate Records**: None found
- **Data Cleaning**: Minimal preprocessing required due to clean dataset

### Preprocessing Steps
1. Removed `loan_id` column (identifier with no predictive value)
2. Label encoded target variable: Approved → 1, Rejected → 0
3. One-hot encoded categorical features: education, self_employed (kept all dummies to maintain feature count)
4. Applied StandardScaler transformation for distance-based models (Logistic Regression, KNN, Naive Bayes)
5. Train-test split with 80-20 ratio using stratified sampling to maintain class distribution

### Final Dataset Properties
- **Training Set**: 3,415 samples (80%)
- **Test Set**: 854 samples (20%)
- **Total Features**: 13 (meets ≥12 requirement)
  - 9 original numerical features
  - 4 dummy variables from categorical encoding
- **Feature Names**: no_of_dependents, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value, education_Graduate, education_Not Graduate, self_employed_No, self_employed_Yes

---

## Models Used

Six classification models were implemented and evaluated on the loan approval dataset:

### 1. Logistic Regression
- **Type**: Traditional Linear Model
- **Configuration**: max_iter=1000, random_state=42
- **Scaling**: Applied StandardScaler

### 2. Decision Tree
- **Type**: Traditional Tree-Based Model
- **Configuration**: max_depth=10, random_state=42
- **Scaling**: Not required

### 3. K-Nearest Neighbors (KNN)
- **Type**: Traditional Instance-Based Model
- **Configuration**: n_neighbors=5
- **Scaling**: Applied StandardScaler

### 4. Naive Bayes
- **Type**: Traditional Probabilistic Model
- **Configuration**: GaussianNB (default parameters)
- **Scaling**: Applied StandardScaler

### 5. Random Forest (Ensemble)
- **Type**: Ensemble Method (Bagging)
- **Configuration**: n_estimators=100, random_state=42
- **Scaling**: Not required

### 6. XGBoost (Ensemble)
- **Type**: Ensemble Method (Boosting)
- **Configuration**: n_estimators=100, random_state=42, eval_metric='logloss'
- **Scaling**: Not required

---

## Model Performance Comparison

### Evaluation Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-------|-----|
| Logistic Regression | 0.9532 | 0.9781 | 0.9621 | 0.9729 | 0.9675 | 0.8919 |
| Decision Tree | 0.9707 | 0.9666 | 0.9779 | 0.9786 | 0.9782 | 0.9314 |
| KNN | 0.9567 | 0.9791 | 0.9651 | 0.9748 | 0.9699 | 0.8993 |
| Naive Bayes | 0.9532 | 0.9794 | 0.9621 | 0.9729 | 0.9675 | 0.8919 |
| Random Forest (Ensemble) | 0.9836 | 0.9958 | 0.9872 | 0.9905 | 0.9889 | 0.9611 |
| XGBoost (Ensemble) | 0.9824 | 0.9957 | 0.9853 | 0.9905 | 0.9879 | 0.9586 |

### Key Findings
- **Best Overall Model**: Random Forest with 98.36% accuracy and F1 score of 0.9889
- **Best AUC Score**: Random Forest (0.9958) showing excellent class separation
- **Top Performers**: Both ensemble methods (Random Forest and XGBoost) significantly outperform traditional models
- **Baseline Performance**: Traditional models achieve 95-97% accuracy, establishing strong baselines
- **Weakest Model**: Naive Bayes and Logistic Regression tied at 95.32% accuracy

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieves solid baseline performance with 95.32% accuracy. Works well because financial features like income, CIBIL score, and asset values have fairly linear relationships with loan approval decisions. The model provides good interpretability through feature coefficients, making it valuable for understanding which factors most influence approval. However, it misses complex non-linear patterns and feature interactions that exist in the data. The balanced precision (0.962) and recall (0.973) indicate no significant bias toward either class. Suitable when model transparency is required for regulatory compliance or stakeholder explanations. |
| Decision Tree | Delivers strong performance at 97.07% accuracy, naturally discovering decision thresholds in the data. The tree structure makes it easy to understand the approval logic - for example, identifying critical CIBIL score cutoffs or income-to-loan ratio thresholds. However, even with max_depth limited to 10, single decision trees remain prone to overfitting and can be unstable with small changes in training data. Different data samples might produce completely different tree structures. The high precision (0.978) and recall (0.979) show excellent balance. Best used when decision interpretability is crucial, though ensemble methods offer better stability. |
| KNN | Reaches 95.67% accuracy by finding the 5 most similar historical loan applications and using their outcomes to predict new applications. The approach makes intuitive sense - applicants with similar financial profiles should have similar approval outcomes. However, performance is heavily dependent on proper feature scaling (which was applied). The model struggles with the curse of dimensionality given 13 features, making true similarity harder to determine in high-dimensional space. Also computationally expensive at prediction time since it must search through all training data. No explicit model is learned, just storing all training examples. Works reasonably but not competitive with ensemble methods for production deployment. |
| Naive Bayes | Achieves 95.32% accuracy with very fast training and prediction times. The probabilistic approach using Bayes theorem is mathematically elegant and works reasonably well for continuous financial features. However, the fundamental assumption of feature independence is clearly violated in loan data - income and asset values are naturally correlated, education level relates to income, loan amount depends on income, etc. This broken assumption limits the model's ability to capture real-world relationships. The identical scores to Logistic Regression (both at 95.32%) is interesting. Best suited when speed is critical, though accuracy suffers compared to methods that can model feature dependencies. |
| Random Forest (Ensemble) | Top performer with 98.36% accuracy and F1 score of 0.9889, demonstrating the power of ensemble learning. By training 100 decision trees on random subsets of data and features, then averaging their predictions, it reduces overfitting while maintaining the ability to capture complex patterns. Excellent at modeling non-linear relationships and feature interactions (like CIBIL score combined with income-to-asset ratios). The outstanding AUC of 0.9958 shows near-perfect ranking ability across all probability thresholds. High precision (0.987) and recall (0.991) mean minimal false positives and false negatives. Provides feature importance scores for interpretability while maintaining robust predictions. The random sampling during training makes it resistant to outliers and noise. Ideal choice for production deployment in loan approval systems. |
| XGBoost (Ensemble) | Nearly matches Random Forest with 98.24% accuracy and F1 of 0.9879, using a different ensemble strategy. Instead of averaging independent trees, XGBoost builds trees sequentially where each new tree focuses on correcting errors from previous trees. This gradient boosting approach is highly effective at learning complex patterns through iterative refinement. Strong performance across all metrics with MCC of 0.959 indicating excellent balance between both classes. Built-in regularization prevents overfitting despite the sequential learning process. Handles the 62:38 class imbalance well through optimization. The model learns subtle patterns in CIBIL scores, income-to-loan ratios, and asset portfolios that simpler models miss. Offers feature importance and fast prediction times through parallelization. Either Random Forest or XGBoost would be excellent choices for deployment, with XGBoost having a slight edge in interpretability of the boosting process. |

---

## Repository Structure

```
loan-approval-prediction/
├── README.md                           # This file
├── ML_Assignment_2.ipynb               # Training notebook with all models
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── loan_approval_dataset.csv           # Original dataset
├── model_comparison_metrics.csv        # Exported evaluation metrics
├── test_data_sample.csv                # Sample test data for demo
└── loan_models_complete.pkl            # All 6 trained models + scaler
```



## Streamlit Application Features

The web application provides:

1. **CSV Upload** : Upload test data in CSV format for batch predictions
2. **Model Selection** : Dropdown menu to select from all 6 trained models
3. **Metrics Display** : Real-time display of all 6 evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
4. **Confusion Matrix** : Visual heatmap showing true positives, true negatives, false positives, and false negatives, plus detailed classification report

Additional features:
- Model type filtering (Ensemble vs Traditional)
- Feature importance visualization for tree-based models
- Interactive metric comparison charts
- Prediction confidence scores
- Performance summary dashboard

---

## Technologies Used

- **Programming Language**: Python 3.8+
- **Machine Learning**: scikit-learn 1.4.0, XGBoost 2.0.3
- **Web Framework**: Streamlit 1.31.0
- **Data Processing**: Pandas 2.1.4, NumPy 1.26.3
- **Visualization**: Matplotlib 3.8.2, Seaborn 0.13.1
- **Model Serialization**: Joblib 1.3.2

---

## Results Summary

- **Best Model**: Random Forest (98.36% accuracy, F1: 0.9889)
- **Runner-up**: XGBoost (98.24% accuracy, F1: 0.9879)
- **Baseline**: Logistic Regression (95.32% accuracy)
- **Key Insight**: Ensemble methods outperform traditional models by ~3-4% in accuracy
- **Recommendation**: Deploy Random Forest for production use due to superior performance and stability

---

## Links

- **GitHub Repository**: https://github.com/rahuldombar/ML_Assignment2
- **Live Streamlit App**: https://mlassignment2-bfbxvsz9swecmyieqhvuhj.streamlit.app/
- **Dataset Source**: https://www.kaggle.com/datasets/vishavgupta01/loan-approval
