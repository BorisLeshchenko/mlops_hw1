import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

def load_params(params_path="params.yaml"):
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_data():
    """Подготовка данных: сплит train/test, сохранение"""
    
    # Загружаем параметры
    params = load_params()
    prepare_params = params['prepare']
    
    print("Загружаем сырые данные...")
    df = pd.read_csv("data/raw/iris.csv")
    print(f"   Загружено {len(df)} строк")
    
    # Целевая переменная
    X = df.drop('target', axis=1)
    y = df['target']
    
    print("Делим на train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=1-prepare_params['split_ratio'],
        random_state=prepare_params['random_state']
    )
    
    # Сохраняем обработанные данные
    os.makedirs("data/processed", exist_ok=True)
    
    joblib.dump(X_train, "data/processed/X_train.pkl")
    joblib.dump(X_test, "data/processed/X_test.pkl")
    joblib.dump(y_train, "data/processed/y_train.pkl")
    joblib.dump(y_test, "data/processed/y_test.pkl")
    
    print("Сохранено:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print("Стадия prepare завершена!")

if __name__ == "__main__":
    prepare_data()