# %% [markdown]
# Data Cleaning: Deep Learning Framework for Option Pricing
# 
# Author: Kyle Rytand Bistrain
# 
# Date: September 2025

# %% [markdown]
# Install Packages
# Python 3.12.4

# %%
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yfinance as yf
import pandas_market_calendars as mcal
from scipy.stats import norm, skew, kurtosis

# %% [markdown]
# Convert simple yields to continuous compounded rates

# %%
def convert_simple_yield_to_continuous(simple_yield, T):
    return np.log(1 + simple_yield*T)/T

# %% [markdown]
# Read in 3-Month Risk Free Interest Rate from FRED

# %%
rf3 = pd.read_csv(
    '/DGS3MO.csv',
    na_values='.',
    parse_dates=['DATE']           
)
rf3['DGS3MO'] = rf3['DGS3MO']/100
rf3['DGS3MO'] = convert_simple_yield_to_continuous(rf3['DGS3MO'], 0.25)
print(rf3.head())
print(rf3[rf3['DGS3MO'].isna()])

# %% [markdown]
# Visualization of 3 Month Interest Rate

# %%
from plotnine import ggplot, aes, geom_line, geom_rug, scale_x_datetime, labs, theme_bw, theme, element_text

p = (
    ggplot(rf3, aes('DATE','DGS3MO'))
    + geom_line(size=0.7)
    + geom_rug(
        data=rf3[rf3['DGS3MO'].isna()],
        sides='b', length=0.02,
        color='#D55E00', alpha=0.8
    )
    + scale_x_datetime(date_breaks='1 year', date_labels='%Y')
    + labs(
         subtitle='',
         x='Date', y='Rate (%)',
         caption='Orange ticks mark missing observations\n\nSource: FRED'
      )
    + theme_bw()
    + theme(
         axis_text_x=element_text(rotation=45, hjust=1),
         figure_size=(8,4)
      )
)

p

# %% [markdown]
# Read in 6-Month Risk Free Interest Rate from FRED

# %%
rf6 = pd.read_csv(
    '/DGS6MO.csv',
    na_values='.',
    parse_dates=['observation_date']           
)
rf6['DGS6MO'] = rf6['DGS6MO']/100
rf6['DGS6MO'] = convert_simple_yield_to_continuous(rf6['DGS6MO'], 0.5)
print(rf6.head())
print(rf6[rf6['DGS6MO'].isna()])

# %% [markdown]
# Visualization of 6 Month Interest Rate

# %%

p = (
    ggplot(rf6, aes('observation_date','DGS6MO'))
    + geom_line(size=0.7)
    + geom_rug(
        data=rf6[rf6['DGS6MO'].isna()],
        sides='b', length=0.02,
        color='#D55E00', alpha=0.8
    )
    + scale_x_datetime(date_breaks='1 year', date_labels='%Y')
    + labs(
         x='Date', y='Rate (%)',
         caption='Orange ticks mark missing observations\n\nSource: FRED'
      )
    + theme_bw()
    + theme(
         axis_text_x=element_text(rotation=45, hjust=1),
         figure_size=(8,4)
      )
)

p

# %% [markdown]
# Read in 1 year Risk Free Interest Rate from FRED

# %%
rf1 = pd.read_csv(
    '/DGS1.csv',
    na_values='.',
    parse_dates=['observation_date']           
)
rf1['DGS1'] = rf1['DGS1']/100
rf1['DGS1'] = convert_simple_yield_to_continuous(rf1['DGS1'], 1)
print(rf1.head())
print(rf1[rf1['DGS1'].isna()])

# %% [markdown]
# Visualization of 1 Year Interest Rate

# %%
from plotnine import ggplot, aes, geom_line, geom_rug, scale_x_datetime, labs, theme_bw, theme, element_text

p = (
    ggplot(rf1, aes('observation_date','DGS1'))
    + geom_line(size=0.7)
    + geom_rug(
        data=rf1[rf1['DGS1'].isna()],
        sides='b', length=0.02,
        color='#D55E00', alpha=0.8
    )
    + scale_x_datetime(date_breaks='1 year', date_labels='%Y')
    + labs(
         x='Date', y='Rate (%)',
         caption='Orange ticks mark missing observations\n\nSource: FRED'
      )
    + theme_bw()
    + theme(
         axis_text_x=element_text(rotation=45, hjust=1),
         figure_size=(8,4)
      )
)

p

# %% [markdown]
# Read in SPY Option Dataset from Delta Neutral (historicaloptiondata.com)

# %%
def read_all_csv_files(directory_path):
    
    all_data = []
    
    
    for filename in sorted(os.listdir(directory_path)):
        
        if filename.startswith("SPY_") and filename.endswith(".csv") and "to" in filename:
            file_path = os.path.join(directory_path, filename)
            print(f"Reading file: {file_path}")
            df = pd.read_csv(file_path,na_values=[".", "", "NA","NaN", "N/A", "?"])
            all_data.append(df)

    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    return combined_data

directory_path = '/optiondata/'

combined_data = read_all_csv_files(directory_path)

print(combined_data.head())


# %%
combined_data['optionroot'].nunique()

# %%
combined_data['expiration'] = pd.to_datetime(combined_data['expiration'], format='%m/%d/%Y')
combined_data['quotedate'] = pd.to_datetime(combined_data['quotedate'], format='%m/%d/%Y')

# %% [markdown]
# Vizualization of SPY Prices from 2012-2024

# %%
from scipy.stats import lognorm

def apply_theme_bw():
    plt.rcParams.update({
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1,
        'grid.color': 'black',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12
    })

apply_theme_bw()

# # Group by quotedate to take the last price
# df_daily_prices = combined_data.groupby('quotedate')['underlying_last'].last().reset_index()

# # Calculate simple returns
# df_daily_prices['simple_return'] = df_daily_prices['underlying_last'].pct_change()

# # Calculate Log returns
# df_daily_prices['log_return'] = np.log(df_daily_prices['underlying_last']).diff()


spy_prices = yf.download(
    "SPY",
    start="2012-02-01",
    end="2024-05-31",
    auto_adjust=False,
    progress=False
)

spy_prices.columns = spy_prices.columns.droplevel('Ticker')

# Compute log returns
spy_prices["logret"] = np.log(spy_prices["Adj Close"] / spy_prices["Adj Close"].shift(1))

# Plotting the Price Distribution

plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
# Distribution of Prices
sns.histplot(spy_prices['Close'], kde=False, bins = 30, color='skyblue')
#plt.title('Distribution of End of Day SPY ETF Prices', fontsize = 12)
plt.xlabel('SPY ETF Price', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.grid(True, alpha=0.3)
plt.plot()


# %% [markdown]
# Log Returns Distribution without Dividends from 2012-2024

# %%

def apply_theme_bw():
    plt.rcParams.update({
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1,
        'grid.color': 'black',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12
    })

apply_theme_bw()

log_returns = spy_prices['logret'].dropna()

print(log_returns)

mu = log_returns.mean()
sigma = log_returns.std()
data_skewness = skew(log_returns)
data_kurtosis = kurtosis(log_returns)

# Generate values for normal PDF
x = np.linspace(log_returns.min(), log_returns.max(), 500)
pdf = norm.pdf(x, loc=mu, scale=sigma)

# Plot the normal PDF
plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'r-', label='Normal PDF', linewidth=2)

sns.histplot(log_returns, kde=False, color='skyblue', stat='density', bins=1000, label='Log Return Data')
plt.axvline(mu, color='green', linestyle='--', linewidth=1.5, label=f'Mean = {mu:.4f}')
plt.axvline(log_returns.median(), color='blue', linestyle='--', linewidth=1.5, label=f'Median = {log_returns.median():.4f}')
plt.text(
    0.02, 0.85,  # Adjust position to avoid clash (move it down)
    f"Skewness = {data_skewness:.4f}\nKurtosis = {data_kurtosis:.4f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8)
)

#plt.title('SPY ETF Log Return Distribution with Fitted Normal PDF', fontsize=14, fontweight='bold', pad=20)  # Add padding
plt.xlabel('Log Returns', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()


# %%
combined_data.head(1000)

# %% [markdown]
# Compute Number of trading days in a year (Optional)

# %%
# from datetime import datetime
# import pandas as pd
# import pandas_market_calendars as mcal
# nyse = mcal.get_calendar('NYSE')
# records = []
# for year in range(2012, 2025):
#     start = f"{year}-01-01"
#     end   = f"{year}-12-31"
#     # valid_days returns a DatetimeIndex of all market‐open timestamps
#     days_index = nyse.valid_days(start_date=start, end_date=end)
    
#     records.append({
#         "year": year,
#         "trading_days": days_index,
#         "count": len(days_index)
#     })
# counts_df = pd.DataFrame([
#     {"year": rec["year"], "n_trading_days": rec["count"]}
#     for rec in records
# ])
# print(counts_df)

# %% [markdown]
# Compute trading days until expiry
# 
# T = # trading days / 252 NYSE trading days

# %%
combined_data['quotedate']  = pd.to_datetime(combined_data['quotedate']).dt.date
combined_data['expiration'] = pd.to_datetime(combined_data['expiration']).dt.date

nyse = mcal.get_calendar('NYSE')

pairs = (
    combined_data
    .loc[:, ['quotedate','expiration']]
    .drop_duplicates()
    .reset_index(drop=True)
)

def count_td_to_expiry(row):
    days = nyse.valid_days(
        start_date=row['quotedate'].isoformat(),
        end_date  =row['expiration'].isoformat()
    )
    return len(days) - 1

pairs['td_to_expiry'] = pairs.apply(count_td_to_expiry, axis=1)

combined_data = combined_data.merge(
    pairs,
    on=['quotedate','expiration'],
    how='left'
)

combined_data['T'] = combined_data['td_to_expiry'] / 252

print("Still missing td_to_expiry:", combined_data['td_to_expiry'].isna().sum())


# %% [markdown]
# Checks Number for Trading Days between dates (Optional)

# %%
# nyse.valid_days(start_date='2012-02-01', end_date='2012-03-15').value_counts().sum()


# %% [markdown]
# Checks Alignment between SPY data from Historical Option Data versus Yahoo Finance (Optional)

# %%
# yf_df = yf.download(
#     "SPY",
#     start=combined_data['quotedate'].min().strftime("%Y-%m-%d"),
#     end=  (combined_data['quotedate'].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
#     auto_adjust=False
# )

# if isinstance(yf_df.columns, pd.MultiIndex):
#     # multi-ticker style: (metric, ticker)
#     adj = yf_df[("Close","SPY")]
# else:
#     # flat style (rare with auto_adjust, but just in case)
#     adj = yf_df["Close"]

# adj.name = "y_adj"
# comp = (
#     combined_data
#       .set_index("quotedate")[["underlying_last"]]
#       .join(adj, how="inner")
#       .dropna(subset=["underlying_last","y_adj"])
# )

# comp["match"] = np.isclose(comp["underlying_last"], comp["y_adj"], atol=1e-8)

# n_total    = len(comp)
# n_matching = comp["match"].sum()
# print(f"{n_matching}/{n_total} ({100*n_matching/n_total:.2f}%) match")


# %%
spy_prices = yf.download(
    "SPY",
    start="2009-01-01",
    end="2024-05-31",
    auto_adjust=False,
    progress=False
)

spy_prices.columns = spy_prices.columns.droplevel('Ticker')

# Compute log returns
spy_prices["logret"] = np.log(spy_prices["Adj Close"] / spy_prices["Adj Close"].shift(1))

# Move date from index to a column for the merge
spy_prices = spy_prices.reset_index()
spy_prices["Date"] = pd.to_datetime(spy_prices["Date"]).dt.tz_localize(None)
combined_data['quotedate'] = pd.to_datetime(combined_data['quotedate'])

# Calculate Realized Volatility 
windows = [63, 126, 252]
vol_cols = []
for w in windows:
    col_name = f"logvol{w}_annual"
    vol_cols.append(col_name)
    daily_std = spy_prices["logret"].rolling(window=w).std(ddof=1)
    spy_prices[col_name] = daily_std * np.sqrt(252)


# Merge Volatility into Main DataFrame 
cols_to_merge = ["Date"] + vol_cols
combined_data = combined_data.merge(
    spy_prices[cols_to_merge],
    how="left",
    left_on="quotedate",
    right_on="Date"
).drop(columns=["Date"])


# Handle Missing Values
# Forward-fill NaNs from holidays and weekends
combined_data[vol_cols] = combined_data[vol_cols].fillna(method='ffill')

print(combined_data.tail())
print(combined_data.isnull().sum())

# %%
# df = yf.download(
#     "SPY",
#     start="2010-01-01",
#     end="2024-05-31",
#     auto_adjust=False
# )
# df["logret"] = np.log(df["Close"] / df["Close"].shift(1))

# # Define your look-back windows (in trading days)
# windows = [63, 126, 252]

# for w in windows:
#     #rolling sample std of the daily returns (ddof=1 by default)
#     daily_std = df["logret"].rolling(window=w).std()

#     # annualize: multiply by sqrt(N/252)
#     df[f"logvol{w}_annual"] = daily_std * np.sqrt(w / 252.0)

# # Map into your combined_data by quotedate
# for w in windows:
#     col = f"logvol{w}_annual"
#     combined_data[col] = combined_data["quotedate"].map(df[col])


# # Sanity check: how many NaNs remain?
# print(combined_data[[f"logvol{w}_annual" for w in windows]].isnull().sum())


# %%
missing_counts = combined_data.isnull().sum()
print("Missing Value Counts per Column:\n", missing_counts)

# %%
combined_data.head(1000)

# %% [markdown]
# Read In Dividend Data and create is_ex_date and days_until_next_exdate

# %%
# Load and clean the dividend data
div = (
    pd.read_csv('/SPY_DIVIDENDS.csv', 
                parse_dates=['PAYABLE DATE', 'RECORD DATE', 'EX-DATE'])
      .query("TICKER == 'SPY'")
      .loc[:, ['DIVIDEND ($)', 'PAYABLE DATE', 'RECORD DATE', 'EX-DATE']]
      .reset_index(drop=True)
)

# Calculate the trailing twelve-month (TTM) dividend amount
div = div.sort_values('EX-DATE')
div['rolling_dividend'] = div['DIVIDEND ($)'].rolling(window=4).sum()

# Prepare and merge the dividend data with the main price data
# sort values
combined_data['quotedate'] = pd.to_datetime(combined_data['quotedate'])
combined_data.sort_values('quotedate', inplace=True)

combined_data_d = pd.merge_asof(
    combined_data,
    div[['EX-DATE', 'rolling_dividend']],
    left_on='quotedate',
    right_on='EX-DATE',
    direction='backward'
)

# Calculate the simple dividend yield (y)
combined_data_d['dividend_yield'] = combined_data_d['rolling_dividend'] / combined_data_d['underlying_last']

# Convert to a continuously compounded rate (q)
combined_data_d['continuous_dividend_rate'] = np.log(1 + combined_data_d['dividend_yield'])


combined_data_d['is_ex_date'] = (
    combined_data_d['quotedate'] == combined_data_d['EX-DATE']
).astype(int)

combined_data_d['next_ex_date'] = pd.merge_asof(
    combined_data,
    div,
    left_on='quotedate',
    right_on='EX-DATE',
    direction='forward'
)['EX-DATE']


# Calculate Trading Days Until Next Ex-Date 

nyse = mcal.get_calendar('NYSE')
nyse_holidays = nyse.holidays().holidays

mask = combined_data_d['next_ex_date'].notna()

start_dates = combined_data_d.loc[mask, 'quotedate'].values.astype('datetime64[D]')
end_dates = combined_data_d.loc[mask, 'next_ex_date'].values.astype('datetime64[D]')

trading_days = np.busday_count(start_dates, end_dates, holidays=nyse_holidays)

combined_data_d['days_until_ex_date'] = np.nan
combined_data_d.loc[mask, 'days_until_ex_date'] = trading_days


print(combined_data_d[['quotedate','underlying_last', 'rolling_dividend', 'dividend_yield', 'continuous_dividend_rate', 'is_ex_date','days_until_ex_date']].head())


# %% [markdown]
# Filter Time to Expiry

# %%
# allow a small window around 60 - 66 for 3 months, 123 to 129 for 6 month, 249 to 255 for 1 year

filtered_df3 = combined_data_d[
    combined_data_d['td_to_expiry'].between(60, 66)
]
filtered_df6 = combined_data_d[
    combined_data_d['td_to_expiry'].between(123, 129)
]

filtered_df1 = combined_data_d[
    combined_data_d['td_to_expiry'].between(249, 255)
]

# %% [markdown]
# Read in VIX

# %%
VIX3m = yf.download(
    "^VIX3M",
    start="2012-02-01",
    end="2024-05-31",
    auto_adjust=True  
)

VIX3m.columns = VIX3m.columns.droplevel(1)  # Flatten the column names
VIX3m.reset_index(inplace=True)           # Make Date a column instead of index

vix_3m_close = VIX3m[["Date", "Close"]]      # Extract Date and Close columns

print(vix_3m_close.head(10))

vix_3m_close['Close'].isna().sum()

# %%
VIX6m = yf.download(
    "^VIX6M",
    start="2012-02-01",
    end="2024-05-31",
    auto_adjust=True  
)

VIX6m.columns = VIX6m.columns.droplevel(1)  # Flatten the column names
VIX6m.reset_index(inplace=True)           # Make Date a column instead of index

vix_6m_close = VIX6m[["Date", "Close"]]      # Extract Date and Close columns

print(vix_6m_close.head(10))
vix_6m_close['Close'].isna().sum()

# %%
filtered_df3['quotedate'] = pd.to_datetime(filtered_df3['quotedate'], errors='coerce')
rf3['DATE'] = pd.to_datetime(rf3['DATE'], errors='coerce')
vix_3m_close['Date'] = pd.to_datetime(vix_3m_close['Date'], errors='coerce')

print("NaT values in 'quotedate' column:", filtered_df3['quotedate'].isna().sum())
print("NaT values in 'DATE' column:", rf3['DATE'].isna().sum())
print("NaT values in 'DATE' column:", vix_3m_close['Date'].isna().sum())

merged_df3 = filtered_df3.merge(rf3, how='left', left_on='quotedate', right_on='DATE')
merged_df3 = merged_df3.merge(vix_3m_close, how='left', left_on='quotedate', right_on='Date')
merged_df3.head()

# %%
filtered_df6['quotedate'] = pd.to_datetime(filtered_df6['quotedate'], errors='coerce')
rf6['observation_date'] = pd.to_datetime(rf6['observation_date'], errors='coerce')
vix_6m_close['Date'] = pd.to_datetime(vix_6m_close['Date'], errors='coerce')

print("NaT values in 'quotedate' column:", filtered_df6['quotedate'].isna().sum())
print("NaT values in 'DATE' column:", rf6['observation_date'].isna().sum())
print("NaT values in 'DATE' column:", vix_6m_close['Date'].isna().sum())

merged_df6 = filtered_df6.merge(rf6, how='left', left_on='quotedate', right_on='observation_date')
merged_df6 = merged_df6.merge(vix_6m_close, how='left', left_on='quotedate', right_on='Date')


merged_df6.head()

# %%
filtered_df1['quotedate'] = pd.to_datetime(filtered_df1['quotedate'], errors='coerce')
rf1['observation_date'] = pd.to_datetime(rf1['observation_date'], errors='coerce')
vix_6m_close['Date'] = pd.to_datetime(vix_6m_close['Date'], errors='coerce')

print("NaT values in 'quotedate' column:", filtered_df1['quotedate'].isna().sum())
print("NaT values in 'DATE' column:", rf1['observation_date'].isna().sum())
print("NaT values in 'DATE' column:", vix_6m_close['Date'].isna().sum())

merged_df1 = filtered_df1.merge(rf1, how='left', left_on='quotedate', right_on='observation_date')
merged_df1 = merged_df1.merge(vix_6m_close, how='left', left_on='quotedate', right_on='Date')
merged_df1.head()

# %%
call_data3 = merged_df3[(merged_df3['type'] == 'call')]
put_data3 = merged_df3[(merged_df3['type'] == 'put')]
print(merged_df3[merged_df3['DGS3MO'].isna()])

# %%
call_data6 = merged_df6[(merged_df6['type'] == 'call')]
put_data6 = merged_df6[(merged_df6['type'] == 'put')]
print(merged_df6[merged_df6['DGS6MO'].isna()])

# %%
call_data1 = merged_df1[(merged_df1['type'] == 'call')]
put_data1 = merged_df1[(merged_df1['type'] == 'put')]
print(merged_df1[merged_df1['DGS1'].isna()])

# %% [markdown]
# Export to CSV

# %%
print(merged_df3.size)
print(call_data3.size)
print(put_data3.size)

print(merged_df6.size)
print(call_data6.size)
print(put_data6.size)

print(merged_df1.size)
print(call_data1.size)
print(put_data1.size)

merged_df3.to_csv("merged_df3.csv", index=False)
merged_df6.to_csv("merged_df6.csv", index=False)
merged_df1.to_csv("merged_df1.csv", index=False)


