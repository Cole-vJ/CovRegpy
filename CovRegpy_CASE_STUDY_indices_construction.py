
# Case Study - construction of 11 sector indices

import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt

sns.set(style='darkgrid')

# pulling the market cap information
financial_data = pd.read_csv('S&P500_Data/constituents.csv', header=0)
tickers = [f"{np.asarray(financial_data)[i, 0].replace('.', '-')}" for i in range(np.shape(financial_data)[0])]
financial_data['Symbol'] = tickers
financial_data = financial_data.set_index('Symbol')
unique_sectors = financial_data[["Sector"]].values
unique_sectors = np.unique(unique_sectors)
financial_data['Shares Outstanding'] = np.zeros(505)
for symbol in tickers:
    print(symbol)
    try:
        financial_data['Shares Outstanding'].loc[symbol] = yf.Ticker(symbol).info['sharesOutstanding']
    except:
        print(symbol + ' shares not available')
        financial_data['Shares Outstanding'].loc[symbol] = np.nan
# financial_data.to_csv('S&P500_Data/sp_500_market_cap.csv')

financial_data = pd.read_csv('S&P500_Data/sp_500_market_cap.csv', header=0)
financial_data = financial_data.set_index('Symbol')
# https://www.nasdaq.com/market-activity/stocks/apa/institutional-holdings
financial_data['Shares Outstanding'].loc['APA'] = 363 * 1e6
# https://www.nasdaq.com/market-activity/stocks/nwl/institutional-holdings
financial_data['Shares Outstanding'].loc['NWL'] = 425 * 1e6
# https://www.nasdaq.com/market-activity/stocks/sbux/institutional-holdings
financial_data['Shares Outstanding'].loc['SBUX'] = 1173 * 1e6
# https://www.nasdaq.com/market-activity/stocks/v/institutional-holdings
financial_data['Shares Outstanding'].loc['V'] = 1667 * 1e6

# pull five years worth of data
# 'RCL' not working for some reason
tickers = pd.read_csv('S&P500_Data/constituents.csv', header=0)
tickers_format = [f"{np.asarray(tickers)[i, 0].replace('.', '-')}" for i in range(np.shape(tickers)[0])]
data = yf.download(tickers_format, start="2016-12-30", end="2022-01-04")
close_data = data['Close']
date_index = pd.date_range(start='31/12/2016', end='01/01/2022')
close_data = close_data.reindex(date_index).interpolate()
close_data = close_data[::-1].interpolate()
close_data = close_data[::-1]
# close_data.to_csv('S&P500_Data/sp_500_close_5_year.csv')

# call RCl individually afterwards and fill
close_data = pd.read_csv('S&P500_Data/sp_500_close_5_year.csv', header=0)
close_data = close_data.set_index('Unnamed: 0')
rcl = yf.download(['RCL'], start="2016-12-30", end="2022-01-04")
rcl_close = rcl['Close']
date_index = pd.date_range(start='31/12/2016', end='01/01/2022')
rcl_close = rcl_close.reindex(date_index).interpolate()
rcl_close = rcl_close[::-1].interpolate()
rcl_close = rcl_close[::-1]
close_data['RCL'] = rcl_close
# close_data.to_csv('S&P500_Data/sp_500_close_5_year.csv')

close_data = pd.read_csv('S&P500_Data/sp_500_close_5_year.csv', header=0)
close_data = close_data.set_index('Unnamed: 0')

# calculate historical market cap data
market_cap_data = close_data.copy()
for symbol in market_cap_data.columns:
    market_cap_data[symbol] = market_cap_data[symbol] * financial_data['Shares Outstanding'].loc[symbol]
# market_cap_data.to_csv('S&P500_Data/sp_500_market_cap_5_year.csv')

market_cap_data = pd.read_csv('S&P500_Data/sp_500_market_cap_5_year.csv', header=0)
market_cap_data = market_cap_data.set_index('Unnamed: 0')

# separate 11 sectors worth of data
unique_sectors = financial_data[["Sector"]].values
unique_sectors = np.unique(unique_sectors)

communication_services_close_data = pd.DataFrame()
communication_services_market_cap_data = pd.DataFrame()
consumer_discretionary_close_data = pd.DataFrame()
consumer_discretionary_market_cap_data = pd.DataFrame()
consumer_staples_close_data = pd.DataFrame()
consumer_staples_market_cap_data = pd.DataFrame()
energy_close_data = pd.DataFrame()
energy_market_cap_data = pd.DataFrame()
financials_close_data = pd.DataFrame()
financials_market_cap_data = pd.DataFrame()
health_care_close_data = pd.DataFrame()
health_care_market_cap_data = pd.DataFrame()
industrials_close_data = pd.DataFrame()
industrials_market_cap_data = pd.DataFrame()
information_technology_close_data = pd.DataFrame()
information_technology_market_cap_data = pd.DataFrame()
materials_close_data = pd.DataFrame()
materials_market_cap_data = pd.DataFrame()
real_estate_close_data = pd.DataFrame()
real_estate_market_cap_data = pd.DataFrame()
utilities_close_data = pd.DataFrame()
utilities_market_cap_data = pd.DataFrame()

for symbol in market_cap_data.columns:
    if financial_data['Sector'].loc[symbol] == 'Communication Services':
        communication_services_close_data[symbol] = close_data[symbol]
        communication_services_market_cap_data[symbol] = market_cap_data[symbol]
    elif financial_data['Sector'].loc[symbol] == 'Consumer Discretionary':
        consumer_discretionary_close_data[symbol] = close_data[symbol]
        consumer_discretionary_market_cap_data[symbol] = market_cap_data[symbol]
    elif financial_data['Sector'].loc[symbol] == 'Consumer Staples':
        consumer_staples_close_data[symbol] = close_data[symbol]
        consumer_staples_market_cap_data[symbol] = market_cap_data[symbol]
    elif financial_data['Sector'].loc[symbol] == 'Energy':
        energy_close_data[symbol] = close_data[symbol]
        energy_market_cap_data[symbol] = market_cap_data[symbol]
    elif financial_data['Sector'].loc[symbol] == 'Financials':
        financials_close_data[symbol] = close_data[symbol]
        financials_market_cap_data[symbol] = market_cap_data[symbol]
    elif financial_data['Sector'].loc[symbol] == 'Health Care':
        health_care_close_data[symbol] = close_data[symbol]
        health_care_market_cap_data[symbol] = market_cap_data[symbol]
    elif financial_data['Sector'].loc[symbol] == 'Industrials':
        industrials_close_data[symbol] = close_data[symbol]
        industrials_market_cap_data[symbol] = market_cap_data[symbol]
    elif financial_data['Sector'].loc[symbol] == 'Information Technology':
        information_technology_close_data[symbol] = close_data[symbol]
        information_technology_market_cap_data[symbol] = market_cap_data[symbol]
    elif financial_data['Sector'].loc[symbol] == 'Materials':
        materials_close_data[symbol] = close_data[symbol]
        materials_market_cap_data[symbol] = market_cap_data[symbol]
    elif financial_data['Sector'].loc[symbol] == 'Real Estate':
        real_estate_close_data[symbol] = close_data[symbol]
        real_estate_market_cap_data[symbol] = market_cap_data[symbol]
    elif financial_data['Sector'].loc[symbol] == 'Utilities':
        utilities_close_data[symbol] = close_data[symbol]
        utilities_market_cap_data[symbol] = market_cap_data[symbol]

communication_services_weights = pd.DataFrame(columns=communication_services_close_data.columns)
communication_services_returns = np.log(communication_services_close_data.iloc[1:].values /
                                        communication_services_close_data.iloc[:-1].values)
consumer_discretionary_weights = pd.DataFrame(columns=consumer_discretionary_close_data.columns)
consumer_discretionary_returns = np.log(consumer_discretionary_close_data.iloc[1:].values /
                                        consumer_discretionary_close_data.iloc[:-1].values)
consumer_staples_weights = pd.DataFrame(columns=consumer_staples_close_data.columns)
consumer_staples_returns = np.log(consumer_staples_close_data.iloc[1:].values /
                                  consumer_staples_close_data.iloc[:-1].values)
energy_weights = pd.DataFrame(columns=energy_close_data.columns)
energy_returns = np.log(energy_close_data.iloc[1:].values /
                        energy_close_data.iloc[:-1].values)
financials_weights = pd.DataFrame(columns=financials_close_data.columns)
financials_returns = np.log(financials_close_data.iloc[1:].values /
                            financials_close_data.iloc[:-1].values)
health_care_weights = pd.DataFrame(columns=health_care_close_data.columns)
health_care_returns = np.log(health_care_close_data.iloc[1:].values /
                             health_care_close_data.iloc[:-1].values)
industrials_weights = pd.DataFrame(columns=industrials_close_data.columns)
industrials_returns = np.log(industrials_close_data.iloc[1:].values /
                             industrials_close_data.iloc[:-1].values)
information_technology_weights = pd.DataFrame(columns=information_technology_close_data.columns)
information_technology_returns = np.log(information_technology_close_data.iloc[1:].values /
                                        information_technology_close_data.iloc[:-1].values)
materials_weights = pd.DataFrame(columns=materials_close_data.columns)
materials_returns = np.log(materials_close_data.iloc[1:].values /
                           materials_close_data.iloc[:-1].values)
real_estate_weights = pd.DataFrame(columns=real_estate_close_data.columns)
real_estate_returns = np.log(real_estate_close_data.iloc[1:].values /
                             real_estate_close_data.iloc[:-1].values)
utilities_weights = pd.DataFrame(columns=utilities_close_data.columns)
utilities_returns = np.log(utilities_close_data.iloc[1:].values /
                           utilities_close_data.iloc[:-1].values)

for date, _ in communication_services_close_data.iterrows():
    if date[-2:] == '01':
        weights_communication_services = communication_services_market_cap_data.loc[date_prev] / \
                                         sum(communication_services_market_cap_data.loc[date_prev])
        weights_consumer_discretionary = consumer_discretionary_market_cap_data.loc[date_prev] / \
                                         sum(consumer_discretionary_market_cap_data.loc[date_prev])
        weights_consumer_staples = consumer_staples_market_cap_data.loc[date_prev] / \
                                   sum(consumer_staples_market_cap_data.loc[date_prev])
        weights_energy = energy_market_cap_data.loc[date_prev] / \
                         sum(energy_market_cap_data.loc[date_prev])
        weights_financials = financials_market_cap_data.loc[date_prev] / \
                             sum(financials_market_cap_data.loc[date_prev])
        weights_health_care = health_care_market_cap_data.loc[date_prev] / \
                              sum(health_care_market_cap_data.loc[date_prev])
        weights_industrials = industrials_market_cap_data.loc[date_prev] / \
                              sum(industrials_market_cap_data.loc[date_prev])
        weights_information_technology = information_technology_market_cap_data.loc[date_prev] / \
                                         sum(information_technology_market_cap_data.loc[date_prev])
        weights_materials = materials_market_cap_data.loc[date_prev] / \
                            sum(materials_market_cap_data.loc[date_prev])
        weights_real_estate = real_estate_market_cap_data.loc[date_prev] / \
                              sum(real_estate_market_cap_data.loc[date_prev])
        weights_utilities = utilities_market_cap_data.loc[date_prev] / \
                            sum(utilities_market_cap_data.loc[date_prev])

    try:
        communication_services_weights.loc[date_prev] = weights_communication_services
        consumer_discretionary_weights.loc[date_prev] = weights_consumer_discretionary
        consumer_staples_weights.loc[date_prev] = weights_consumer_staples
        energy_weights.loc[date_prev] = weights_energy
        financials_weights.loc[date_prev] = weights_financials
        health_care_weights.loc[date_prev] = weights_health_care
        industrials_weights.loc[date_prev] = weights_industrials
        information_technology_weights.loc[date_prev]= weights_information_technology
        materials_weights.loc[date_prev] = weights_materials
        real_estate_weights.loc[date_prev] = weights_real_estate
        utilities_weights.loc[date_prev] = weights_utilities

    except:
        pass
    date_prev = date

sector_indices = pd.DataFrame(columns=['Communication Services', 'Consumer Discretionary', 'Consumer Staples',
                                       'Energy', 'Financials', 'Health Care', 'Industrials', 'Information Technology',
                                       'Materials', 'Real Estate', 'Utilities'],
                              index=close_data.index[1:])
sector_indices['Communication Services'] = \
    np.sum(communication_services_weights * communication_services_returns, axis=1)
sector_indices['Consumer Discretionary'] = \
    np.sum(consumer_discretionary_weights * consumer_discretionary_returns, axis=1)
sector_indices['Consumer Staples'] = \
    np.sum(consumer_staples_weights * consumer_staples_returns, axis=1)
sector_indices['Energy'] = \
    np.sum(energy_weights * energy_returns, axis=1)
sector_indices['Financials'] = \
    np.sum(financials_weights * financials_returns, axis=1)
sector_indices['Health Care'] = \
    np.sum(health_care_weights * health_care_returns, axis=1)
sector_indices['Industrials'] = \
    np.sum(industrials_weights * industrials_returns, axis=1)
sector_indices['Information Technology'] = \
    np.sum(information_technology_weights * information_technology_returns, axis=1)
sector_indices['Materials'] = \
    np.sum(materials_weights * materials_returns, axis=1)
sector_indices['Real Estate'] = \
    np.sum(real_estate_weights * real_estate_returns, axis=1)
sector_indices['Utilities'] = \
    np.sum(utilities_weights * utilities_returns, axis=1)

sector_indices = sector_indices.drop(['2022-01-01'])

# sector_indices.to_csv('S&P500_Data/sp_500_11_sector_indices.csv')

# load 11 sector indices
sector_11_indices = pd.read_csv('S&P500_Data/sp_500_11_sector_indices.csv', header=0)
sector_11_indices = sector_11_indices.set_index(['Unnamed: 0'])

# approximate daily treasury par yield curve rates for 3 year bonds
risk_free = (0.01 / 365)  # daily risk free rate

# sector numpy array
sector_11_indices_array = np.vstack((np.zeros((1, 11)), np.asarray(sector_11_indices)))

for col, sector in enumerate(sector_11_indices.columns):
    plt.plot(np.asarray(np.cumprod(np.exp(sector_11_indices_array[:, col]))), label=sector)
plt.title(textwrap.fill('Cumulative Returns of Eleven Market Cap Weighted Sector Indices of S&P 500 from 1 January 2017 to 31 December 2021', 60),
          fontsize=10)
plt.legend(loc='upper left', fontsize=8)
plt.xticks([0, 365, 730, 1095, 1461, 1826],
           ['31-12-2016', '31-12-2017', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5'], fontsize=8)
plt.show()
