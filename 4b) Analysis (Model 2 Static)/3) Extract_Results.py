import os
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb


def extract_feature_importance(model_filename, X_train, X_test, output_folder, subfolder):
    # Load the model
    model = joblib.load(model_filename)

    # Ensure the 'models' directory exists within the subfolder
    models_directory = os.path.join(output_folder, subfolder)
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)  # Create the directory if it doesn't exist

    # Check the model type and extract feature importance or coefficients
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor, xgb.XGBRegressor)):
        print(f"Extracting feature importances for {model_filename}...")
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        feature_importances = np.abs(shap_values.values).mean(axis=0)
        influence_direction = np.sign(shap_values.values.mean(axis=0))
        feature_summary = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': feature_importances,
            'Direction': influence_direction
        }).sort_values(by='Importance', ascending=False)

        # Save the feature importances in the correct directory
        feature_summary.to_csv(
            os.path.join(models_directory, f'{os.path.basename(model_filename)}_feature_importances.csv'), index=False)

    elif isinstance(model, LinearRegression):
        print(f"Extracting coefficients for {model_filename}...")
        coef = model.coef_
        coef_summary = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': np.abs(coef),
            'Direction': np.sign(coef)
        }).sort_values(by='Importance', ascending=False)

        # Save the coefficients in the correct directory
        coef_summary.to_csv(os.path.join(models_directory, f'{os.path.basename(model_filename)}_coefficients.csv'),
                            index=False)

    elif isinstance(model, MLPRegressor):
        print(f"Using SHAP for Neural Network ({model_filename})...")
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(X_test)
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        influence_direction = np.sign(shap_values.values.mean(axis=0))

        shap_summary = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': feature_importance,
            'Direction': influence_direction
        }).sort_values(by='Importance', ascending=False)
        shap_summary.to_csv(os.path.join(models_directory, f'{os.path.basename(model_filename)}_shap_values.csv'),
                            index=False)

    else:
        print(f"Model {model_filename} does not support feature importance extraction.")


def process_top_models_comparison(model_comparison_csv, X_train, X_test):
    # Read the model performance comparison
    results_df = pd.read_csv(model_comparison_csv)

    # Select the top 3 models based on R2 score (for regression tasks)
    top_models_rows = results_df.head(1)  # Get the top 3 models based on the comparison results
    print(f"Top models: {top_models_rows[['Model', 'R2 Score']]}")

    output_folder = '2b) Model_Results'
    subfolder = 'top_models_results'  # New subfolder for storing the results
    print(f"Results will be saved in: {os.path.join(output_folder, subfolder)}")

    # Loop through the top 3 models
    for _, row in top_models_rows.iterrows():
        best_model_filename = f"{row['Model'].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.pkl"
        print(f"Processing Model: {best_model_filename}")

        model_path = os.path.join(output_folder, 'models', best_model_filename)
        print(f"Model Path: {model_path}")  # Add this to check the final model path

        # Extract feature importance or SHAP values
        extract_feature_importance(model_path, X_train, X_test, output_folder, subfolder)


if __name__ == "__main__":
    # Example usage
    model_comparison_csv = '2b) Model_Results/model_comparison_results.csv'

    X_train = pd.read_csv('2b) Model_Results/datasets/X_train.csv')
    X_test = pd.read_csv('2b) Model_Results/datasets/X_test.csv')
    y_train = pd.read_csv('2b) Model_Results/datasets/y_train.csv')
    y_test = pd.read_csv('2b) Model_Results/datasets/y_test.csv')

    process_top_models_comparison(model_comparison_csv, X_train, X_test)
