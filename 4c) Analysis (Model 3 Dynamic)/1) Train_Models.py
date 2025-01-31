import os
import pandas as pd
import shap
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import time  # To measure calculation time


def extract_feature_importance(model, model_name, X, y, folder_name):
    os.makedirs(folder_name, exist_ok=True)

    if model_name == 'Linear Regression':
        for idx, target in enumerate(y.columns):
            coef = model.estimators_[idx].coef_
            importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': np.abs(coef),
                'Direction': np.sign(coef)
            }).sort_values(by='Importance', ascending=False)
            importance_filename = os.path.join(folder_name, f'{target}_coefficients_sorted.csv')
            importances.to_csv(importance_filename, index=False)

    elif model_name in ['Random Forest', 'XGBoost']:
        for idx, target in enumerate(y.columns):
            explainer = shap.Explainer(model.estimators_[idx], X)
            shap_values = explainer(X)
            feature_importance = np.abs(shap_values.values).mean(axis=0)
            influence_direction = np.sign(shap_values.values.mean(axis=0))

            importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importance,
                'Direction': influence_direction
            }).sort_values(by='Importance', ascending=False)
            importance_filename = os.path.join(folder_name, f'{target}_importances_sorted.csv')
            importances.to_csv(importance_filename, index=False)


def main():
    df = pd.read_csv(
        '/Users/sanaemessoudi/Desktop/DSMA/Final_Project_DSMA_Final/2c) Data_Model_2/4b) Data_with_Revenue.csv')
    X = df.drop(columns=['revenue_Monday', 'revenue_Tuesday', 'revenue_Wednesday', 'revenue_Thursday',
                         'revenue_Friday', 'revenue_Saturday', 'revenue_Sunday'])
    y = df[['revenue_Monday', 'revenue_Tuesday', 'revenue_Wednesday', 'revenue_Thursday',
            'revenue_Friday', 'revenue_Saturday', 'revenue_Sunday']]

    X.fillna(X.mean(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # XGBoost parameters (removed early_stopping_rounds)
    xgb_params = {
        'n_estimators': 1000,  # Number of trees
        'learning_rate': 0.05,  # Slower learning to prevent overfitting
        'max_depth': 6,  # A moderate depth to prevent overfitting
        'min_child_weight': 1,  # Prevents trees from growing too deep
        'subsample': 0.8,  # Sampling ratio for each tree
        'colsample_bytree': 0.8,  # Sampling ratio for each feature
        'gamma': 0.1,  # Minimum loss reduction to make a further partition
        'reg_alpha': 0.1,  # L1 regularization term
        'reg_lambda': 0.1,  # L2 regularization term
        'objective': 'reg:squarederror',  # Regression problem
        'n_jobs': -1,  # Use all CPUs
    }

    models = [
        ('Linear Regression', MultiOutputRegressor(LinearRegression())),
        ('Random Forest', MultiOutputRegressor(RandomForestRegressor())),
        ('XGBoost', MultiOutputRegressor(xgb.XGBRegressor(**xgb_params)))
    ]

    best_model = None
    best_model_name = ""
    best_score = float('inf')
    performance_metrics = []

    for model_name, model in models:
        start_time = time.time()  # Start the timer
        print(f"Training model: {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        elapsed_time = time.time() - start_time  # End the timer and calculate elapsed time

        print(f"Mean Squared Error for {model_name}: {mse}")
        print(f"R² Score for {model_name}: {r2}")
        print(f"Training time for {model_name}: {elapsed_time} seconds")

        performance_metrics.append({
            'Model': model_name,
            'MSE': mse,
            'R²': r2,
            'Time (s)': elapsed_time
        })

        if mse < best_score:
            best_score = mse
            best_model = model
            best_model_name = model_name

    print(f"\nBest model: {best_model_name} with MSE: {best_score}")
    model_filename = '1b) best_revenue_prediction_model.pkl'
    joblib.dump(best_model, model_filename)

    scaler_filename = '1c) scaler.pkl'
    joblib.dump(scaler, scaler_filename)

    folder_name = '1d) feature_importances'
    extract_feature_importance(best_model, best_model_name, X, y, folder_name)

    print(f"Model and scaler have been saved to {model_filename} and {scaler_filename}")
    print(
        f"Feature importances or coefficients for each target variable have been saved to the '{folder_name}' folder.")

    # Save performance metrics to a CSV file
    performance_df = pd.DataFrame(performance_metrics)
    performance_df.to_csv('model_performance.csv', index=False)
    print("\nModel performance metrics have been saved to 'model_performance.csv'")


if __name__ == "__main__":
    main()