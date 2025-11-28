import yaml
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import os

def load_params(params_path="params.yaml"):
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def train_model():
    """Обучение модели с логированием в MLflow"""
    
    # Загружаем параметры
    params = load_params()
    train_params = params['train']
    
    print("Загружаем обработанные данные...")
    X_train = joblib.load("data/processed/X_train.pkl")
    X_test = joblib.load("data/processed/X_test.pkl")
    y_train = joblib.load("data/processed/y_train.pkl")
    y_test = joblib.load("data/processed/y_test.pkl")
    
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Настраиваем MLflow
    mlflow.set_experiment("iris_classification")
    with mlflow.start_run():
        
        # Логируем параметры
        mlflow.log_params(train_params)
        mlflow.log_param("dataset_size", len(X_train) + len(X_test))
        
        print("Обучаем модель...")
        model = LogisticRegression(
            random_state=train_params['random_state'],
            max_iter=train_params['max_iter']
        )
        model.fit(X_train, y_train)
        
        # Предсказания и метрики
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        
        # Логируем метрики
        mlflow.log_metric("accuracy", accuracy)
        
        # Сохраняем модель
        model_path = "models/model.pkl"
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        print("Модель сохранена: models/model.pkl")
        print("Стадия train завершена!")
        
        return model, accuracy

if __name__ == "__main__":
    train_model()