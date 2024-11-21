from matplotlib import pyplot as plt
from roc_function2 import train_and_get
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import numpy as np

# Load datasets and call the function
df = pd.read_csv('/Users/student/Desktop/ANON36_HF_EKG_output.csv')
# change variable names
df = df.rename(columns={
    'Risk- Hypertension' : 'Hypertension',
    'Risk- Hypercholesteremia' : 'Hypercholesteremia', 
    'Risk- Hypertriglyceridemia' : 'Hypertriglyceridemia',
    'Risk- High HDL' : 'High HDL', 
    'Risk- Low HDL' : 'Low HDL', 
    'Risk- Diabetes' : 'Diabetes', 
    'Risk- Family Hx' : 'Family Hx',
    'Risk- Smoke' : 'Smoke', 
    'ASCVD PCE Risk Score' :'ASCVD PCE Risk',
    'PREVENT Score - ASCVD 10-year risk' : 'PREVENT ASCVD Risk', 
    'CVD PREVENT 10yr' : 'PREVENT CVD Risk',
    'HF PREVENT 10-year' : 'PREVENT HF Risk', 
    'Total A' : 'Total CAC Score', 
    'LM A' : 'CAC Score (Left Main)', 
    'LAD A' : 'CAC Score (LAD)', 
    'Cx' : 'CAC Score (LCx)', 
    'RCA' : 'CAC Score (RCA)',
    'Num Lesions' : 'Number of Total Lesions', 
    'Lesion Vol Sum' : 'Total Volume Score', 
    'Lesion Max D' : 'Peak CAC Density', 
    'Lesion Ave D' : 'Mean CAC Density', 
    'Male' : 'Gender',
    'Race_encoded' : 'Race',
})
# df = df[df['Gender'] == 1]
X_columns = df.columns.difference(['CONFIRM ID', 'result'], sort=False)
X_columns = list(X_columns)
y_column = 'result'
categorical_columns = [
  'Hypertension', 'Hypercholesteremia', 'Hypertriglyceridemia', 'High HDL', 'Low HDL', 'Diabetes', 'Family Hx','Smoke', 'Gender'
]

"""
# Define the columns you want to check for missing values
columns_to_check = ['Total CAC Score', 'ASCVD PCE Risk', 'PREVENT HF Risk']
# Filter the dataset to exclude rows with missing values in these columns
df_filtered = df.dropna(subset=columns_to_check)
"""
# Call the function for different datasets and specify the output paths
fpr1, tpr1, roc_auc1, label1 = train_and_get(df, X_columns, y_column, 'date_after_ML', categorical_columns=categorical_columns)
fpr2, tpr2, roc_auc2, label2 = train_and_get(df, ['Total CAC Score'], y_column, 'date_after_CAC_Total')
fpr3, tpr3, roc_auc3, label3 = train_and_get(df, ['ASCVD PCE Risk'], y_column, 'date_after_PCE')
fpr4, tpr4, roc_auc4, label4 = train_and_get(df, ['PREVENT HF Risk'], y_column, 'date_after_PREVENT_HF')

# Plot all ROC curves in one figure
plt.figure(figsize=(10, 8))
plt.plot(fpr1, tpr1, lw=2, label=f'{label1} (AUC = {roc_auc1:.3f})', color='blue')
plt.plot(fpr2, tpr2, lw=2, label=f'{label2} (AUC = {roc_auc2:.3f})', color='red')
plt.plot(fpr3, tpr3, lw=2, label=f'{label3} (AUC = {roc_auc3:.3f})', color='green')
plt.plot(fpr4, tpr4, lw=2, label=f'{label4} (AUC = {roc_auc4:.3f})', color='orange')

# Add diagonal line for random chance
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

# Plot formatting
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate / 1 - Specificity')
plt.ylabel('True Positive Rate / Sensitivity')
plt.title('ROC Curve HF')
plt.legend(loc="lower right")
plt.grid(True)

# Show the plot
plt.show()