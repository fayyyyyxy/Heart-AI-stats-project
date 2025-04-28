import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np

def train_and_get(df, X_columns, y_column, dataset_label, categorical_columns=None):
    """
    Function to train a model and get the ROC curve data.
    
    Parameters:
    df_train (pd.DataFrame): The dataset to train the model.
    X_columns (list): List of feature columns to use for training and prediction.
    y_column (str): The target column in both df_train.
    dataset_label (str): Label for the dataset, used for identifying the ROC curve.
    categorical_columns (list): List of columns that should be treated as categorical. Default is None.
    
    Returns:
    fpr (array): False positive rate for ROC curve.
    tpr (array): True positive rate for ROC curve.
    roc_auc (float): AUC score.
    dataset_label (str): The label for the dataset (passed in).
    """
    # Select the columns for X and y
    X = df[X_columns].copy()  # Use the provided X_columns
    y = df[y_column]          # Use the provided y_column

    # Make specified columns categorical (if provided)
    if categorical_columns is not None:
        for col in categorical_columns:
            if col in X.columns:
                # Convert the specified columns to 'category' type for both training and prediction data
                X[col] = X[col].astype('category')

    # Initialize StratifiedKFold for 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=94)

    # Create the XGBClassifier
    clf = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=42)

    # Placeholder for the stacked probabilities on the prediction set
    stacked_probabilities = np.zeros(y.shape)

    # Cross-validation loop
    for train_index, test_index in cv.split(X, y):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        # Train the model
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
        # Predict probabilities 
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        stacked_probabilities[test_index] = y_pred_proba

    # Calculate ROC curve and AUC 
    fpr, tpr, _ = roc_curve(y, stacked_probabilities)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc, dataset_label