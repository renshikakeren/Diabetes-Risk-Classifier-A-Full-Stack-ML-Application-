import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from data_preprocessing import load_and_preprocess_data


def train_and_save_model():
    """
    Trains multiple models, finds the best one using GridSearchCV, and saves it.
    """
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('data/diabetes.csv')

    if X_train is None:
        return

    # --- Hyperparameter Tuning with GridSearchCV for Multiple Models ---

    models_to_tune = {
        'RandomForest': (RandomForestClassifier(random_state=42), {
            'n_estimators': [100, 200],
            'max_depth': [10, 20]
        }),
        'SVM': (SVC(probability=True, random_state=42), {
            'C': [0.1, 1],
            'kernel': ['rbf']
        }),
        'XGBoost': (XGBClassifier(random_state=42), {
            'n_estimators': [100, 200],
            'max_depth': [3, 5]
        }),
        'LogisticRegression': (LogisticRegression(random_state=42), {
            'C': [0.1, 1, 10],
            'solver': ['liblinear']
        })
    }

    best_models = {}
    best_f1_score = 0.0
    best_model_name = ""

    for name, (model, params) in models_to_tune.items():
        print(f"\nTraining and tuning {name}...")
        grid_search = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # Evaluate on the test set
        y_pred = grid_search.best_estimator_.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        best_models[name] = {
            'model': grid_search.best_estimator_,
            'f1_score': f1,
            'best_params': grid_search.best_params_
        }
        print(f"Best {name} F1 Score: {f1:.4f}")

        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = name

    print(f"\n--- Best Model is {best_model_name} with F1 Score: {best_f1_score:.4f} ---")

    # Save the best-performing model and the scaler object
    joblib.dump(best_models[best_model_name]['model'], 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    print("Best model and scaler saved successfully to the 'models/' directory.")


if __name__ == '__main__':
    # Make sure you have a `data/diabetes.csv` file before running this.
    train_and_save_model()
