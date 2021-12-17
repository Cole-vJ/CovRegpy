
import numpy as np
import pandas as pd
import yfinance as yf

from AdvEMDpy import AdvEMDpy, emd_basis

# seed random number generation
np.random.seed(0)

# pull all close data
tickers_format = ['ABG.JO', 'AGL.JO', 'AMS.JO', 'ANG.JO', 'APN.JO', 'BHP.JO', 'BID.JO', 'BTI.JO', 'BVT.JO', 'CFR.JO',
                  'CLS.JO', 'CPI.JO', 'DSY.JO', 'EXX.JO', 'FSR.JO', 'GFI.JO', 'GLN.JO', 'GRT.JO', 'IMP.JO', 'INL.JO',
                  'INP.JO', 'MCG.JO', 'MNP.JO', 'MRP.JO', 'MTN.JO', 'NED.JO', 'NPH.JO', 'NPN.JO', 'NRP.JO', 'OMU.JO',
                  'PRX.JO', 'REM.JO', 'RNI.JO', 'SBK.JO', 'SHP.JO', 'SLM.JO', 'SOL.JO', 'SPP.JO', 'SSW.JO', 'VOD.JO',
                  'WHL.JO']
data = yf.download(tickers_format, start="2018-12-31", end="2022-01-01")
close_data = data['Close']
del data, tickers_format

# create date range and interpolate
date_index = pd.date_range(start='31/12/2018', end='01/01/2022')
close_data = close_data.reindex(date_index).interpolate()
close_data = close_data[::-1].interpolate()
close_data = close_data[::-1]
del date_index

# daily risk free rate
risk_free = (0.02 / 365)

# setup time and knots for EMD
time = np.arange(np.shape(close_data)[0])
knots = 70

# calculate returns for CRC
returns = (np.log(np.asarray(close_data)[1:, :]) -
           np.log(np.asarray(close_data)[:-1, :]))

# store tickers and partition model days and forecast days
tickers = close_data.columns.values.tolist()
model_days = 731  # 2 years - less a month
forecast_days = np.shape(close_data)[0] - model_days - 30

# set up basis for mean extraction in CRC
spline_basis_transform = emd_basis.Basis(time_series=np.arange(model_days), time=np.arange(model_days))
spline_basis_transform = spline_basis_transform.cubic_b_spline(knots=np.linspace(0, model_days - 1, knots))

# store weights calculated throughout model
weights = np.zeros((forecast_days, np.shape(close_data)[1]))


