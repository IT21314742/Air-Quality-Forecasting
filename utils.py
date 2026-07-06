"""
Air quality Forecasting - Utility Functions
Beijing PM2.5 hourly air quality dataset (UCI). Target: pm2.5 (µg/m³, continuous).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Data Loading and Preprocessing

def load_data(filepath="data/PRSA_data.csv"):
    """
    load the rawBeijing PM2.5 hourly dataset and build a proper datetime index.
    
    The raw file stores the timestamp across four integer columns (year, month, day, hour). We combine them into a single 'Datetime' column so pandas can
    treat the data  as an ordered  time targets. 
    """
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df[["year","month","day", "hour"]])
    return df

# Feature Engineering

def create_features(df):
    """
    Engineer calendar, lag, and rolling features from the sorted time series.
    
    Calendar features capture deterministic seasonality (time-of-the-day, day-of-the-week,
    month, etc.). Lag and rolling features gives the model recent historical context
    without leaking future information - Lags are shifted by the lag length so every
    row only looks backward. The categorical wind direction 'cbwd' is one-hot
    encoded.
    
    Parameters
    ----------
    df: pd.Dataframe with 'Datetime' (datetime64), 'pm2.5' (target) and the raw
        weather columns, sorted ascending by Datetime.
    
    Returns
    ----------
    pd.dataframe with new feature columns; rows with NaN target or NaN lag/rolling
    warmup values are dropped.
    
    """
    
    df = df.copy().sort_values("Datetime").reset_index(drop=True)
    
    
    # Calender features
    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek
    df["month"] = df["Datetime"].dt.month
    df["quarter"] = df["Datetime"].dt.quarter
    df["year"] = df["Datetime"].dt.year
    df["dayofyear"] = df["Datetime"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    
    # Lag features - shift by lag length to avoid data leakage
    df["lag_1"] = df["pm2.5"].shift(1)      # previous hour
    df["lag_2"] = df["pm2.5"].shift(24)     # same hour, 1 day ago
    
    # Rolling mean features - window ends 1 period before the current row
    df["roll_24"] = df["pm2.5"].shift(1).rolling(window=24).mean()
    df["roll_168"] = df["pm2.5"]