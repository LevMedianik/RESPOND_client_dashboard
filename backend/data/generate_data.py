import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Финишная дата
END_DATE = datetime(2025, 9, 30, 23, 0, 0)  # до конца дня
START_DATE = END_DATE - timedelta(days=3*365)  # примерно 3 года назад
SEED = 42
np.random.seed(SEED)

DATA_DIR = Path(__file__).resolve().parent
HOURLY_PATH = DATA_DIR / "respond_hourly.csv"
MONTHLY_PATH = DATA_DIR / "respond.csv"

def generate_data():
    # Почасовой диапазон
    hours = pd.date_range(start=START_DATE, end=END_DATE, freq="H")

    n = len(hours)
    t = np.arange(n)

    # --- Leads (заявки в час) ---
    base_leads = 5
    trend = np.linspace(0, 10, n)                         # общий рост
    daily_cycle = 2 * np.sin(2 * np.pi * (t % 24) / 24)   # активность по часам
    weekly_cycle = 3 * np.sin(2 * np.pi * (t % (24*7)) / (24*7))  # недельный цикл
    noise = np.random.normal(0, 1.5, n)
    leads = base_leads + trend/500 + daily_cycle + weekly_cycle + noise
    leads = np.clip(leads.round().astype(int), 0, None)  # заявки не могут быть отрицательными

    # --- CPL (стоимость лида) ---
    base_cpl = 30
    monthly_cycle = 5 * np.cos(2 * np.pi * t / (24*30))  # месячный цикл
    noise_cpl = np.random.normal(0, 2, n)
    cpl = base_cpl + monthly_cycle + noise_cpl
    cpl = np.clip(cpl.round(2), 5, None)

    # --- ROI ---
    base_roi = 0.3
    seasonal_roi = 0.1 * np.sin(2 * np.pi * t / (24*365))  # годовой цикл
    noise_roi = np.random.normal(0, 0.05, n)
    roi = base_roi + seasonal_roi + noise_roi
    roi = np.clip(roi.round(3), -0.5, 1.2)  # допускаем отрицательный ROI

    # --- Почасовой DataFrame ---
    df_hourly = pd.DataFrame({
        "datetime": hours,
        "leads": leads,
        "cpl": cpl,
        "roi": roi
    })
    df_hourly.to_csv(HOURLY_PATH, index=False)

    # --- Агрегация по месяцам ---
    df_hourly["month"] = df_hourly["datetime"].dt.to_period("M")
    df_monthly = df_hourly.groupby("month").agg({
        "leads": "sum",
        "cpl": "mean",
        "roi": "mean"
    }).reset_index()
    df_monthly["month"] = df_monthly["month"].astype(str)

    # округляем значения
    df_monthly["cpl"] = df_monthly["cpl"].round(2)
    df_monthly["roi"] = df_monthly["roi"].round(3)

    df_monthly.to_csv(MONTHLY_PATH, index=False)

    print(f"Датасеты сохранены:\n - {HOURLY_PATH}\n - {MONTHLY_PATH}")

if __name__ == "__main__":
    generate_data()
