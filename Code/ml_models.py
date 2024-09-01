from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pandas as pd
import numpy as np

class MLModels:
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None

    def prepare_data(self, df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_random_forest(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.rf_model = grid_search.best_estimator_

    def train_xgboost(self, X_train, y_train):
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 1.0]
        }
        xgb_model = xgb.XGBClassifier(random_state=42)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.xgb_model = grid_search.best_estimator_

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv("data/processed/cleaned_product_data.csv")
    ml_models = MLModels()
    X_train, X_test, y_train, y_test = ml_models.prepare_data(data, target_column='high_demand')
    
    ml_models.train_random_forest(X_train, y_train)
    rf_accuracy, rf_report = ml_models.evaluate_model(ml_models.rf_model, X_test, y_test)
    print("Random Forest Performance:", rf_accuracy)
    print(rf_report)

    ml_models.train_xgboost(X_train, y_train)
    xgb_accuracy, xgb_report = ml_models.evaluate_model(ml_models.xgb_model, X_test, y_test)
    print("XGBoost Performance:", xgb_accuracy)
    print(xgb_report)