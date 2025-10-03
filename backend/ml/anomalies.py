import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("backend/data/respond.csv")

def detect_anomalies(metric: str = "cpl", k: float = 2.5):
    """
    Детекция аномалий по Z-score для метрики (CPL или ROI).
    |z| > k считается аномалией.
    
    Parameters
    ----------
    metric : str
        "cpl" или "roi"
    k : float
        порог по модулю для Z-score
    """
    df = pd.read_csv(DATA_PATH)

    if metric not in ["cpl", "roi"]:
        raise ValueError("metric must be 'cpl' or 'roi'")

    values = df[metric].values
    mean, std = np.mean(values), np.std(values)

    z_scores = (values - mean) / std
    anomalies = df[np.abs(z_scores) > k].copy()
    anomalies["z_score"] = z_scores[np.abs(z_scores) > k]

    return anomalies[["month", metric, "z_score"]]

if __name__ == "__main__":
    # Пример использования
    anomalies_cpl = detect_anomalies(metric="cpl", k=2.5)
    anomalies_roi = detect_anomalies(metric="roi", k=2.5)

    print("Аномалии CPL:")
    print(anomalies_cpl if not anomalies_cpl.empty else "Не найдено")

    print("\nАномалии ROI:")
    print(anomalies_roi if not anomalies_roi.empty else "Не найдено")
