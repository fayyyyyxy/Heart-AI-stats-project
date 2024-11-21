from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from ROC_function import train_and_get_roc_curve
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import numpy as np

# Load datasets and call the function
df = pd.read_csv('/Users/student/Desktop/death_withmed.csv')
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
X_columns = df.columns.difference(['CONFIRM ID', 'Death'], sort=False)
X_columns = list(X_columns)
y_column = 'Death'
categorical_columns = [
  'Hypertension', 'Hypercholesteremia', 'Hypertriglyceridemia', 'High HDL', 'Low HDL', 'Diabetes', 'Family Hx','Smoke', 'Gender'
]

# Assuming df_train is your training data with features X_columns and target y_column

# Define the class imbalance ratio
negative_count = (df[y_column] == 0).sum()
positive_count = (df[y_column] == 1).sum()
scale_pos_weight = negative_count / positive_count

# Define the XGBoost classifier
xgb_model = xgb.XGBClassifier(
    eval_metric='logloss',  
    scale_pos_weight=scale_pos_weight
)

# Define the parameter grid for Grid Search
param_grid = {
    'max_depth': [3, 6],         
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200],  
    'subsample': [0.8],          
    'colsample_bytree': [0.8],   
    'gamma': [0.1],              
}

# Set up Stratified K-Fold cross-validation (10 folds)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Set up GridSearchCV with 10-fold cross-validation
grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='f1',  # Use the desired scoring metric
    cv=cv,         # 10-fold cross-validation
    verbose=1,
    n_jobs=-1      # Use all available cores
)

# Perform Grid Search on the entire training dataset (no need to pre-split)
grid_search_xgb.fit(df[X_columns], df[y_column])

# Get the best parameters and estimator
best_xgb_model = grid_search_xgb.best_estimator_
print(f"Best parameters: {grid_search_xgb.best_params_}")
