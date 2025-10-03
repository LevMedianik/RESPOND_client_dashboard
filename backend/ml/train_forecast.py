import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from prophet import Prophet

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.ml.features import make_features

DATA_PATH = Path("backend/data/respond_hourly.csv")
MODEL_PATH = Path("backend/models/forecast.pkl")

def evaluate_model(y_true, y_pred):
    """Метрики RMSE и R²"""
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

def train_forecast(horizon: int = 24):
    # Загружаем данные
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"]).set_index("datetime")

    # Генерируем фичи
    X, y, X_future = make_features(df, horizon=horizon)

    # Разбивка по времени
    tscv = TimeSeriesSplit(n_splits=5)

    # Модели
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "DecisionTree": DecisionTreeRegressor(max_depth=6),
        "RandomForest": RandomForestRegressor(n_estimators=200),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200),
        "GradientBoosting": GradientBoostingRegressor(),
        "LightGBM": lgb.LGBMRegressor(),
        "XGBoost": xgb.XGBRegressor(),
        "Prophet": None  # особый случай
    }

    best_rmse, best_r2 = float("inf"), -float("inf")
    best_model, best_name = None, None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            if name == "Prophet":
                prophet_df = df.reset_index()[["datetime", "leads"]].rename(columns={"datetime": "ds", "leads": "y"})
                prophet = Prophet()
                prophet.fit(prophet_df)
                future = prophet.make_future_dataframe(periods=horizon, freq="H")
                forecast = prophet.predict(future)
                y_pred = forecast["yhat"].iloc[:len(y)]
                rmse, r2 = evaluate_model(y, y_pred)
                mlflow.log_param("model", name)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                model_to_save = prophet
            else:
                rmses, r2s = [], []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    rmse, r2 = evaluate_model(y_val, y_pred)
                    rmses.append(rmse)
                    r2s.append(r2)
                rmse, r2 = np.mean(rmses), np.mean(r2s)

                mlflow.log_param("model", name)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.sklearn.log_model(model, artifact_path="model")
                model_to_save = model

            if (rmse < best_rmse) or (rmse == best_rmse and r2 > best_r2):
                best_rmse, best_r2 = rmse, r2
                best_model, best_name = model_to_save, name

    # Сохраняем лучшую модель
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Лучшая модель: {best_name} (RMSE={best_rmse:.3f}, R²={best_r2:.3f})")
    print(f"Модель сохранена в {MODEL_PATH}")

if __name__ == "__main__":
    train_forecast(horizon=24)
