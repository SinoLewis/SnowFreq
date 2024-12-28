import numpy as np
import scipy.stats as ss
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from FMNM.Processes import Heston_process, VG_process, GARCH
from arch.unitroot import PhillipsPerron, KPSS, ADF
from arch import arch_model
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import pacf
import matplotlib.gridspec as gridspec
from FMNM.probabilities import VG_pdf
from scipy.integrate import quad
from scipy.optimize import minimize

"""
A. LINEAR REGRESSION 

Real market data
  Data Cleaning
Linear regression
The Kalman filter
  Kalman regression for the beta
"""
def data_cleaning():
  filename = "data/" + "historical_data.csv"
  data = pd.read_csv(filename, index_col="Date", parse_dates=True)
  
  print(data.tail())
  # Check for empty & non-positive values
  print(data[data.isna().any(axis=1)].tail())
  print((data < 1e-2).any())
  df = data[["GOOGL", "^GSPC"]]
  
  history_len = 1000  # lenght of the time series
  df.sort_index(inplace=True, ascending=True)  # not necessary in general, but useful
  df = df.dropna(axis=1, how="all")  # drops columns with all NaNs
  df = df.dropna(axis=0, how="all")  # drops rows with at least one NaN
  df = df[-history_len:]
  df = df.ffill()  # Forward fill
  print("Are there still some NaNs? ")
  df.isnull().any()
  
"""
B. Auto-correlation tracking

Autoregressive process AR(1)
  Regression analysis
Kalman filter
Non-constant autocorrelation
  Example 1
  Example 2
"""

"""
C. VOLATILITY TRACKING

Heston Path generation
  Log-return analysis
  Hypothesis testing
  A digression on the Variance Gamma process
Garch(1,1)
  GARCH with the arch library
  GARCH from scratch
  Rolling variance
Kalman filter
  Linear Gaussian State Space Model
  Algorithm implementation
"""