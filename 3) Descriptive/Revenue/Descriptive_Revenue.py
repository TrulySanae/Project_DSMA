import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression


def extract_linear_regression_coefficients(model_filename, X_train, output_folder):
    # Load the model
    model = joblib.load(model_filename)

    # Ensure the model is a Linear Regression model
    if not isinstance(model, LinearRegression):
        print(f"Skipping {model_filename}: Not a Linear Regression model.")
        return

    print(f"Extracting coefficients for {model_filename}...")
    coef = model.coef_  # For linear regression
    coef_summary = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': coef
    }).sort_values(by='Coefficient', ascending=False)

    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the coefficients in the correct directory
    coef_summary.to_csv(os.path.join(output_folder, f'{os.path.basename(model_filename)}_coefficients.csv'),
                        index=False)


if __name__ == "__main__":
    # Example usage
    model_filename = '/Users/sanaemessoudi/Desktop/DSMA/Final_Project_DSMA_Final/4b) Analysis (Model 2 Static)/2b) Model_Results/models/Linear_Regression.pkl'  # Adjust the path as necessary
    output_folder = 'linear_regression_results'

    X_train = pd.read_csv('/Users/sanaemessoudi/Desktop/DSMA/Final_Project_DSMA_Final/4b) Analysis (Model 2 Static)/2b) Model_Results/datasets/X_train.csv')

    extract_linear_regression_coefficients(model_filename, X_train, output_folder)
