import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("backend/data/respond_hourly.csv")

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляем календарные признаки (hour, dow, month + sin/cos кодировки)."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    # sin/cos для циклических признаков
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df

def add_lag_features(df: pd.DataFrame, target: str = "leads", lags=[1, 2, 6, 12, 24]) -> pd.DataFrame:
    """Добавляем лаги по целевой переменной (leads)."""
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, target: str = "leads", windows=[24, 168]) -> pd.DataFrame:
    """Добавляем скользящие средние по часам (сутки и неделя)."""
    df = df.copy()
    for w in windows:
        df[f"{target}_rollmean{w}"] = df[target].shift(1).rolling(window=w).mean()
    return df

def make_features(df: pd.DataFrame, horizon: int = 24):
    """
    Основная функция для генерации признаков.
    Возвращает X, y для обучения и X_future для прогноза.
    """
    df = df.copy()

    # --- базовые фичи ---
    df = add_time_features(df)

    # --- лаги и скользящие ---
    df = add_lag_features(df, target="leads")
    df = add_rolling_features(df, target="leads")

    # --- выбрасываем NA после лагов ---
    df = df.dropna()

    # --- разделение ---
    X = df.drop(columns=["cpl", "roi", "leads"])
    y = df["leads"]

    # --- формируем future X (на будущее прогнозирование) ---
    last_index = df.index[-1]
    future_index = pd.date_range(last_index + pd.Timedelta(hours=1), periods=horizon, freq="H")
    df_future = pd.DataFrame(index=future_index)

    # генерируем те же временные признаки
    df_future = add_time_features(df_future)

    # лаги: берем последние значения из df
    for lag in [1, 2, 6, 12, 24]:
        df_future[f"leads_lag{lag}"] = df["leads"].iloc[-lag]

    # скользящие средние
    for w in [24, 168]:
        df_future[f"leads_rollmean{w}"] = df["leads"].iloc[-w:].mean()

    X_future = df_future

    return X, y, X_future

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"]).set_index("datetime")
    X, y, X_future = make_features(df, horizon=24)
    print("Признаки готовы")
    print("X:", X.shape, "y:", y.shape, "X_future:", X_future.shape)
