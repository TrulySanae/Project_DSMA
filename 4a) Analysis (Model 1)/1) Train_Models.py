import os
import time
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, auc)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
import re  # Regular expression module for cleaning model names
from joblib import Parallel, delayed  # For parallel processing


def gini_coefficient(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return 2 * auc(fpr, tpr) - 1


def top_decile_lift(y_true, y_scores):
    data = pd.DataFrame({'y_true': y_true, 'y_scores': y_scores})
    data = data.sort_values(by='y_scores', ascending=False)
    top_10_pct = int(len(data) * 0.1)
    top_decile = data.iloc[:top_10_pct]
    return (top_decile['y_true'].mean()) / (data['y_true'].mean())


def hit_rate(y_true, y_pred):
    return np.mean(y_true == y_pred)


def train_model(model_name, model, X_train, y_train):
    """Train the model and return the model_name and the trained model"""
    model.fit(X_train, y_train)
    return model_name, model


def get_best_model(path, drop_columns, target_column):
    output_folder = '1b) Model_Results'
    models_folder = os.path.join(output_folder, 'models')
    datasets_folder = os.path.join(output_folder, 'datasets')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(datasets_folder, exist_ok=True)

    df = pd.read_csv(path)
    df = df.drop(drop_columns, axis=1)
    X = df.drop([target_column], axis=1)
    y = df[target_column]

    scaler = StandardScaler()
    X_columns = X.columns
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=X_columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save datasets to CSV
    X_train.to_csv(os.path.join(datasets_folder, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(datasets_folder, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(datasets_folder, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(datasets_folder, 'y_test.csv'), index=False)

    models = {
        'Logistic Regression (liblinear)': LogisticRegression(solver='liblinear', max_iter=1000, penalty='l2'),
        'Logistic Regression (newton-cg)': LogisticRegression(solver='newton-cg', max_iter=1000, penalty='l2'),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Bagging': BaggingClassifier(),
        'Naïve Bayes': GaussianNB(),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss'),
        'Support Vector Machine': SVC(probability=True),
        'Neural Network': MLPClassifier(max_iter=2500)
    }

    param_grids = {
        'Support Vector Machine': {
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'C': [0.1, 1, 10]
        },
        'Neural Network': {
            'hidden_layer_sizes': [
                (100, 50), (50, 25), (100, 50, 25), (150, 100, 50),
                (100, 100, 50, 25), (150, 150, 100, 50)
            ],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam'],
            'alpha': [0.001, 0.01],
            'learning_rate': ['adaptive', 'invscaling'],
            'verbose': [True]
        }
    }

    results = {}
    best_svm_model = None
    best_nn_model = None
    start_time = time.time()

    # Subset the data for faster hyperparameter tuning
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

    # Parallelize training models
    models_trained = Parallel(n_jobs=-1)(
        delayed(train_model)(model_name, model, X_train_subset, y_train_subset)
        for model_name, model in models.items()
    )

    for model_name, model in models_trained:
        model_start_time = time.time()

        if model_name in param_grids:
            param_grid = param_grids[model_name]
            grid_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=5, cv=2, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            model_name += f" ({grid_search.best_params_})"

            if model_name.startswith('Support Vector Machine'):
                best_svm_model = best_model
            if model_name.startswith('Neural Network'):
                best_nn_model = best_model
        else:
            best_model = model
            best_model.fit(X_train, y_train)

        # Clean the model name to create a valid file path
        clean_model_name = re.sub(r'[^\w\s-]', '', model_name.replace(' ', '_'))  # Remove invalid characters
        model_filename = os.path.join(models_folder, f"{clean_model_name}.pkl")
        joblib.dump(best_model, model_filename)

        # Predictions and scoring
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_prob) if hasattr(best_model, 'predict_proba') else 0
        gini = gini_coefficient(y_test, y_prob)
        tdl = top_decile_lift(y_test, y_prob)
        hit = hit_rate(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        model_time = time.time() - model_start_time

        results[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "Gini": gini,
            "TDL": tdl,
            "Hit Rate": hit,
            "Confusion Matrix": conf_matrix,
            "Training Time (s)": model_time
        }

    results_df = pd.DataFrame(results).T.reset_index()
    results_df.rename(columns={'index': 'Model'}, inplace=True)
    results_df = results_df.sort_values(by='ROC AUC', ascending=False)
    results_df.to_csv(os.path.join(output_folder, 'model_comparison_results.csv'), index=False)

    total_time = time.time() - start_time
    print(f"\nModel comparison completed in {total_time:.2f} seconds.")
    return best_svm_model, best_nn_model, results_df


if __name__ == "__main__":
    best_svm, best_nn, performance_df = get_best_model('/Users/sanaemessoudi/Desktop/DSMA/Final_Project_DSMA_Final/2b) Data_Model_1/3b) Supervised_Data_No_Outliers.csv', [], 'rating')
