
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def mean_return(weights, all_returns, window):

    if np.shape(weights)[0] > np.shape(weights)[1]:
        weights = weights.T
    if np.shape(all_returns)[0] > np.shape(all_returns)[1]:
        all_returns = all_returns.T
    portfolio_returns = np.sum(weights * all_returns, axis=0)

    mean_returns = np.zeros(int(np.shape(weights)[1] - window))
    for col in range(len(mean_returns)):
        mean_returns[col] = np.mean(portfolio_returns[col:int(col + window)])

    return mean_returns


def variance_return(weights, all_returns, window):

    if np.shape(weights)[0] > np.shape(weights)[1]:
        weights = weights.T
    if np.shape(all_returns)[0] > np.shape(all_returns)[1]:
        all_returns = all_returns.T
    portfolio_returns = np.sum(weights * all_returns, axis=0)

    variance_returns = np.zeros(int(np.shape(weights)[1] - window))
    for col in range(len(mean_returns)):
        variance_returns[col] = np.var(portfolio_returns[col:int(col + window)])

    return variance_returns


if __name__ == "__main__":

    # pull all close data
    tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
    data = yf.download(tickers_format, start="2018-12-31", end="2021-12-01")
    close_data = data['Close']
    del data, tickers_format

    # create date range and interpolate
    date_index = pd.date_range(start='31/12/2018', end='12/01/2021')
    close_data = close_data.reindex(date_index).interpolate()
    close_data = close_data[::-1].interpolate()
    close_data = close_data[::-1]
    del date_index

    # calculate returns and realised covariance
    returns = (np.log(np.asarray(close_data)[1:, :]) -
               np.log(np.asarray(close_data)[:-1, :]))
    realised_covariance = np.cov(returns.T)
    risk_free = (0.02 / 365)

    stock_weights = np.random.uniform(0, 1, (np.shape(returns)))
    stock_weights = stock_weights.T / np.sum(stock_weights, axis=1)
    window_width = 90

    mean_returns = mean_return(weights=stock_weights, all_returns=returns, window=window_width)

    plt.title(f'Mean Returns with {window_width} Day Window')
    plt.plot(mean_returns)
    plt.show()

    variance_returns = variance_return(weights=stock_weights, all_returns=returns, window=window_width)

    plt.title(f'Variance Returns with {window_width} Day Window')
    plt.plot(variance_returns)
    plt.show()
