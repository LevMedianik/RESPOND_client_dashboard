from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import os
from pathlib import Path
from backend.ml.features import make_features
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

DATA_HOURLY = Path("backend/data/respond_hourly.csv")
DATA_MONTHLY = Path("backend/data/respond.csv")
MODEL_PATH = Path("backend/models/forecast.pkl")

# Инициализация приложения
app = FastAPI(title="RE:SPOND Dashboard API")

# --- Раздача фронтенда ---
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def root():
    """Главная страница — отдаём index.html"""
    return FileResponse(os.path.join(frontend_dir, "index.html"))

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics(n: int = 12):
    """Возвращает последние n месяцев KPI"""
    df = pd.read_csv(DATA_MONTHLY)
    data = df.tail(n).to_dict(orient="records")
    return {"data": data}


@app.get("/forecast")
def forecast():
    """Прогноз по лидам до декабря 2025"""
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Модель не загружена"})

    df = pd.read_csv(DATA_HOURLY, parse_dates=["datetime"]).set_index("datetime")
    last_date = df.index.max()
    end_date = pd.Timestamp("2025-12-31 23:00:00")

    # Prophet или sklearn-like модель
    if "prophet" in str(type(model)).lower():
        from prophet import Prophet
        future = pd.date_range(last_date + pd.Timedelta(hours=1), end_date, freq="H")
        future_df = pd.DataFrame({"ds": future})
        forecast_df = model.predict(future_df)[["ds", "yhat"]].rename(
            columns={"ds": "datetime", "yhat": "leads_forecast"}
        )
    else:
        horizon = int((end_date - last_date).total_seconds() // 3600)
        X, y, X_future = make_features(df, horizon=horizon)
        y_pred = model.predict(X_future)
        forecast_df = pd.DataFrame({"datetime": X_future.index, "leads_forecast": y_pred})

    # --- агрегируем по месяцам ---
    forecast_df["month"] = forecast_df["datetime"].dt.to_period("M")
    df_monthly = forecast_df.groupby("month")["leads_forecast"].sum().reset_index()

    # берём только полные месяцы после last_date
    first_full_month = (last_date + pd.offsets.MonthBegin(1)).to_period("M")
    df_monthly = df_monthly[df_monthly["month"] >= first_full_month]
    df_monthly["month"] = df_monthly["month"].astype(str)

    return {"forecast_monthly": df_monthly.to_dict(orient="records")}


@app.get("/anomalies")
def anomalies(metric: str = "cpl", k: float = 2.5):
    """Детекция аномалий (Z-score)"""
    df = pd.read_csv(DATA_MONTHLY)
    mean, std = df[metric].mean(), df[metric].std()
    df["z_score"] = (df[metric] - mean) / std
    df["month"] = df["month"].astype(str)
    anomalies = df.loc[df["z_score"].abs() > k, ["month", metric, "z_score"]]
    return {"anomalies": anomalies.to_dict(orient="records")}
