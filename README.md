## Project Description

This project contains work related to both machine learning modeling for predicting major cardiovascular outcomes and risk score development from the model’s prediction.  
The outcomes analyzed include Major Adverse Cardiac Events (MACE), Heart Failure (HF), and Death.  
The project involves data preprocessing, model training using XGBoost, performance evaluation, and risk stratification.

---

## Research Affiliation

This project was conducted at **Emory University School of Medicine**, Department of Radiology and Imaging Sciences, under the Translational Laboratory for Cardiothoracic Imaging and Artificial Intelligence.

---

## Dataset Used

> **Note:** The actual datasets are not included in this repository due to privacy and confidentiality restrictions. The following describes the datasets used during the project:

#### ANON36_data
- Anonymized baseline datasets used for outcome prediction.
- Three separate datasets were used, each tailored for predicting MACE, HF, or Death, respectively.

#### ANON36_data_EKG
- Extended versions of ANON36_data incorporating EKG-derived features (e.g., VentricularRate, QTInterval).
- Three corresponding datasets were used, aligned with the baseline data for MACE, HF, and Death.

---

## Folder Descriptions

### Data Preprocessing
- Contains R scripts for selecting and preparing modeling variables, including demographic variables, risk factors, clinical variables, clinical risk scores, medication, and CT image analysis variables.
- Handles data cleaning, imputation, encoding, and feature transformation to generate consistent modeling inputs across all outcomes.

### XGBoost_Feature Importances and ROC
- Visualizes feature importance (gain, weight, cover) from trained XGBoost models.
- Includes ROC plots to connect feature contributions with outcome-specific model performance.

### XGBoost_ROC Comparison
- Compares against baseline models that each relied solely on a single clinical risk score for prediction, highlighting the added value of the full feature set and machine learning approach.

### XGBoost_ROC for each fold
- Contains ROC plots for each cross-validation fold, separately for MACE, HF, and Death models.
- Assesses the stability and generalizability of each model.

### XGBoost_Confusion Matrix
- Contains confusion matrices summarizing classification performance on testing data for all three outcomes.

### XGBoost_SHAP
- Includes SHAP (SHapley Additive exPlanations) plots to explain model predictions globally and at the individual level.

### Develop Risk Score_MACE
- Focused on developing a clinically interpretable risk score framework specifically for MACE.
- Explored multiple strategies for defining risk group cutoffs, including slope detection on the predicted probability distribution curve to identify inflection points, percentile-based thresholds (e.g., 5th, 50th, 95th percentiles), and comparison against existing clinical risk score cutoffs.
- Evaluates and compares the resulting groupings using event rates, group-specific AUC, and weighted AUC.

---
