import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('/Users/student/Desktop/ANON36_death_EKG_output.csv')

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

# Select the columns
X = df.drop(['CONFIRM ID', 'result'], axis=1)
y = df['result']

# Make some columns categorical
categorical_cols = [X.columns[i] for i in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,45]]
for col in categorical_cols:
    X[col] = X[col].astype('category')


# Initialize StratifiedKFold for 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=94)

# Create the XGBClassifier
clf = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=42)

# Placeholder for the stacked probabilities
stacked_probabilities = np.zeros(y.shape)

# List to hold AUC scores for each fold
auc_scores = []

# Cross-validation loop
for train_index, test_index in cv.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train the model
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Predict probabilities
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    stacked_probabilities[test_index] = y_pred_proba
    
    # Calculate AUC for the current fold
    fold_auc = roc_auc_score(y_test, y_pred_proba)
    auc_scores.append(fold_auc)

# Final evaluation
final_auc = roc_auc_score(y, stacked_probabilities)
print(f"Final AUC-ROC: {final_auc}")

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y, stacked_probabilities)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate / 1 - Specificity')
plt.ylabel('True Positive Rate / Sensitivity')
plt.title('Receiver Operating Characteristic (ROC)(Death)')
plt.legend(loc="lower right")
plt.show()

# plot feature importance
# define groups
Clinical_Risk_Scores = ['ASCVD PCE Risk', 'PREVENT ASCVD Risk', 'PREVENT CVD Risk',
       'PREVENT HF Risk']

Image_Analysis = [ 'Total CAC Score', 'CAC Score (Left Main)',
       'CAC Score (LAD)', 'CAC Score (LCx)', 'CAC Score (RCA)',
       'Number of Total Lesions', 'Total Volume Score', 'Peak CAC Density',
       'Mean CAC Density']

Medication = ['Beta Blockers', 'Antianginal Agents',
       'Antihyperlipidemic', 'Calcium Blockers', 'Antihypertensive',
       'Antidiabetic', 'Antiarrhythmic']

EKG = ['VentricularRate', 'AtrialRate', 'QRSDuration',
       'QTInterval', 'QTCorrected', 'RAxis', 'TAxis', 'QRSCount', 'QOnset',
       'QOffset', 'TOffset', 'PRInterval', 'PAxis', 'POnset', 'POffset']

# Get the Booster object from the trained XGBoost model
booster = clf.get_booster()

# Get feature importance
importance_dict = booster.get_score(importance_type='total_gain')

# Convert importance dictionary to a DataFrame for easier manipulation
importances_df = pd.DataFrame(list(importance_dict.items()), columns=['Features', 'Gain'])
importances_df.sort_values(by='Gain', ascending=False, inplace=True)

# Assign colors to each feature based on their group
def assign_color(feature):
    if feature in Clinical_Risk_Scores:
        return 'darkblue'  
    elif feature in Image_Analysis:
        return 'thistle'
    elif feature in Medication:
        return 'grey'
    elif feature in EKG:
        return 'pink'
    else:
        return 'gold'  

importances_df['Color'] = importances_df['Features'].apply(assign_color)

# Plot feature importances with custom colors
plt.barh(importances_df['Features'], importances_df['Gain'], color=importances_df['Color'])
plt.xlabel('Total Gain')
plt.ylabel('Features')
plt.title('Variable Ranking for Prediction of Death')
plt.gca().invert_yaxis()  # Invert y axis for better visibility of top features

plt.show()
