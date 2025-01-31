import os
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


def extract_feature_importance(model_filename, X_train, X_test, output_folder, subfolder):
    # Load the model
    model = joblib.load(model_filename)

    # Ensure the 'models' directory exists within the subfolder
    models_directory = os.path.join(output_folder, subfolder)
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)

    # Scale X_test if the model was trained with scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform X_train
    X_test_scaled = scaler.transform(X_test)  # Transform X_test with the same scaler

    if isinstance(model, LogisticRegression):
        print(f"Extracting coefficients for {model_filename}...")
        coef = model.coef_[0]  # For binary logistic regression
        coef_summary = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': np.abs(coef),  # Absolute value for importance
            'Direction': np.sign(coef)  # Sign for influence direction
        }).sort_values(by='Importance', ascending=False)
        coef_summary.to_csv(os.path.join(models_directory, f'{os.path.basename(model_filename)}_coefficients.csv'),
                            index=False)

    elif isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, MLPClassifier)):
        print(f"Using SHAP for {model_filename}...")
        explainer = shap.Explainer(model, X_train_scaled)
        shap_values = explainer(X_test_scaled)

        feature_importance = np.abs(shap_values.values).mean(axis=0)
        influence_direction = np.sign(shap_values.values.mean(axis=0))

        shap_summary = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': feature_importance,
            'Direction': influence_direction
        }).sort_values(by='Importance', ascending=False)

        shap_summary.to_csv(os.path.join(models_directory, f'{os.path.basename(model_filename)}_shap_importance.csv'),
                            index=False)

    elif isinstance(model, xgb.XGBClassifier):
        print(f"Using SHAP TreeExplainer for {model_filename}...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)

        feature_importance = np.abs(shap_values).mean(axis=0)  # No need for `values` attribute here
        influence_direction = np.sign(shap_values.mean(axis=0))

        shap_summary = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': feature_importance,
            'Direction': influence_direction
        }).sort_values(by='Importance', ascending=False)

        shap_summary.to_csv(os.path.join(models_directory, f'{os.path.basename(model_filename)}_shap_importance.csv'),
                            index=False)

    else:
        print(f"Model {model_filename} does not support feature importance extraction.")


def process_top_models_comparison(model_comparison_csv, X_train, X_test):
    results_df = pd.read_csv(model_comparison_csv)
    top_models_rows = results_df.head(1)

    output_folder = '1b) Model_Results'
    subfolder = 'top_models_results'
    print(f"Results will be saved in: {os.path.join(output_folder, subfolder)}")

    for _, row in top_models_rows.iterrows():
        best_model_filename = f"{row['Model'].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.pkl"
        model_path = os.path.join(output_folder, 'models', best_model_filename)
        print(f"Processing Model: {best_model_filename}")
        extract_feature_importance(model_path, X_train, X_test, output_folder, subfolder)


if __name__ == "__main__":
    model_comparison_csv = '1b) Model_Results/model_comparison_results.csv'
    X_train = pd.read_csv('1b) Model_Results/datasets/X_train.csv')
    X_test = pd.read_csv('1b) Model_Results/datasets/X_test.csv')
    process_top_models_comparison(model_comparison_csv, X_train, X_test)