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
