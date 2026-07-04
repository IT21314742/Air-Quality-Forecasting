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
    treat the data  as an ordered 
    """