# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 09:41:12 2025

@author: HP
"""

import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
import statsmodels.api as sm
from statsmodels.genmod.families.links import logit
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import scipy.optimize as minimize
from calendar import monthrange
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import math
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.linear_model import HuberRegressor
import seaborn as sns
import itertools
from arch.bootstrap import MovingBlockBootstrap

#from sell_in import load_sell_in_az
az_2024 = pd.read_excel("C:\\Users\\HP\\Downloads\\2024 Amazon Data.xlsx",sheet_name="Data")
az_2025 = pd.read_excel("C:\\Users\\HP\\Downloads\\boAt Audio Sellout as on 26th June 2025.xlsx")
fk_prices = pd.read_csv("C:\\Users\\HP\\Downloads\\fk_prices.csv")
fk_prices['date'] = pd.to_datetime(fk_prices['date'])
#az_sellout = pd.read_excel("C:\\Users\\HP\\Downloads\\Optimization\\Amazon Sellout.xlsx",sheet_name="Data")
#az_sellin = load_sell_in_az()
common_cols = az_2024.columns.intersection(az_2025.columns)
az_sellout = pd.concat([az_2024[common_cols], az_2025[common_cols]], ignore_index=True)
az_sellout.fillna(0, inplace=True)
az_sellout['net_ordered_units'] = az_sellout['net_ordered_units'].round(0).astype(int)
sap_codes  = pd.read_excel("C:\\Users\\HP\\Downloads\\ASIN vs PF.xlsx")
traffic = pd.read_excel("C:\\Users\\HP\\Downloads\\traffic_spend_monthly.xlsx")
sap_codes = sap_codes[sap_codes['SAP code']!=0]
sap_codes = sap_codes[['SAP code','asin','PF','Category']]
az_sellout.drop_duplicates(inplace=True,subset=['asin', 'snapshot_day','net_ordered_units'])
cols = ['net_ordered_units', 'net_ordered_gms_amt','sellable_qty', 'sellable_value']
az_sellout[cols] = az_sellout[cols].fillna(0)
az_sellout[cols] = az_sellout[cols].replace(r'^\s*$', np.nan, regex=True)
az_sellout[cols] = az_sellout[cols].astype(float)

# az_sellout = pd.merge(az_sellout, sap_codes, on='asin', how='left')
az_sellout['price'] = np.where(az_sellout['net_ordered_units']==0,0,az_sellout['net_ordered_gms_amt']/az_sellout['net_ordered_units']*1.18)

# product_similarity = pd.read_excel("C:\\Users\\HP\\Downloads\\product_similarity.xlsx")
az_sellout['snapshot_day'] = pd.to_datetime(az_sellout['snapshot_day'])
az_sellout['year'] = az_sellout['snapshot_day'].dt.year
az_sellout['month'] = az_sellout['snapshot_day'].dt.month

az_sellout_daily = az_sellout.loc[:,['week', 'year','month', 'snapshot_day', 'asin',
                                     'further_classification', 'net_ordered_units',
                                     'sellable_qty','price']]
az_sellout_daily = pd.merge(az_sellout_daily,sap_codes,how='inner',on='asin')
az_sellout_daily.columns = ['week', 'year', 'month', 'date', 'asin',
       'further_classification', 'sales', 'sellable_qty','price','sap_code','PF','Category']
az_sellout_daily = pd.merge(az_sellout_daily,fk_prices,how='left',on=['sap_code','date'])
az_sellout_daily = az_sellout_daily.rename({'price_x':'price','price_y':'fk_price'},axis=1)
# window_size=5
# Compute 7-day moving average of sales per asin
# az_sellout_daily['sales_ma5'] = (
#     az_sellout_daily
#     .groupby('asin')['sales']
#     .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
# )
# # Compute 7-day moving average of sales per asin
# az_sellout_daily['sales_ema5'] = (
#     az_sellout_daily
#     .groupby('asin')['sales']
#     .transform(lambda x: x.ewm(span=5, adjust=1).mean())
# )
az_sellout_daily['day'] = az_sellout_daily['date'].dt.day

az_sellout_daily = az_sellout_daily.sort_values(['asin', 'date'])

# az_sellout_daily['sellable_qty_lag1'] = az_sellout_daily.groupby('asin')['sellable_qty'].shift(1)
# az_sellout_daily['sales_lag1'] = az_sellout_daily.groupby('asin')['sales'].shift(1)
def filter_asins(df, min_units=30, min_months=1):
    df = df.copy()

    # Clean negative values
    df.loc[df['sales'] < 0, 'price'] = 0
    df.loc[df['sales'] < 0, 'sales'] = 0

    # Step 1: Filter by sales in the last month
    latest_year = df['year'].max()
    latest_month = df[df['year'] == latest_year]['month'].max()
    last_month_df = df[(df['year'] == latest_year) & (df['month'] == latest_month)]

    asin_sales = last_month_df.groupby('asin')['sales'].sum()
    asins_with_enough_sales = asin_sales[asin_sales >= min_units].index

    # Step 2: Filter by data availability over months
    asin_month_counts = (
        df[['asin', 'year', 'month']]
        .drop_duplicates()
        .groupby('asin')
        .size()
    )
    asins_with_enough_months = asin_month_counts[asin_month_counts >= min_months].index

    # Step 3: Take intersection of both filters
    valid_asins = asins_with_enough_sales.intersection(asins_with_enough_months)

    filtered_df = df[df['asin'].isin(valid_asins)]

    # Step 4: Drop rows before first non-zero price for each asin
    def drop_pre_price(g):
        g = g.sort_values('date')
        first_nonzero_idx = g[g['price'] > 0].first_valid_index()
        if first_nonzero_idx is not None:
            return g.loc[first_nonzero_idx:]
        else:
            return pd.DataFrame(columns=g.columns)  # Drop group if no non-zero price

    filtered_df = (
        filtered_df.groupby('asin', group_keys=False)
        .apply(drop_pre_price)
        .reset_index(drop=True)
    )

    return filtered_df



az_sellout_daily = filter_asins(az_sellout_daily, min_units=30)
# invalid = az_sellout_daily[az_sellout_daily.isna().any(axis=1)].unique()  # Check for any remaining NaNs

# az_sellout[az_sellout['asin'].isin(invalid)][['asin','item_name']].drop_duplicates()

def price_changes(df, k=2, m=0.05):
    """
    Identify significant price changes for each ASIN based on deviation from the last key price.
    """
    df = df.copy()
    df = df.sort_values(['asin', 'date'])

    # Work only with positive prices
    filtered = df[df['price'] > 0].copy()

    # Prepare output columns
    filtered['price_changed'] = 0
    filtered['perc_change'] = np.nan
    filtered['last_key_price'] = np.nan

    # Process each ASIN separately
    for asin, group in filtered.groupby('asin'):
        prices = group['price'].values
        dates = group['date'].values
        perc_changes = []
        key_flags = []
        last_key_price = None
        std_estimate = None

        # Estimate std from naive pct_change to use as a threshold
        temp_pct = pd.Series(prices).pct_change().dropna()
        std_estimate = temp_pct.std() if not temp_pct.empty else 0

        for i, price in enumerate(prices):
            if last_key_price is None:
                # First price is always a change
                perc_changes.append(0)
                key_flags.append(1)
                last_key_price = price
            else:
                pct = (price - last_key_price) / last_key_price if last_key_price != 0 else 0
                condition = abs(pct) > k * std_estimate or abs(pct) > m
                if condition:
                    key_flags.append(1)
                    last_key_price = price
                else:
                    key_flags.append(0)
                perc_changes.append(pct)

        # Assign back
        filtered.loc[group.index, 'perc_change'] = perc_changes
        filtered.loc[group.index, 'price_changed'] = key_flags
    cleaned_blocks = []
    for code_num, group in filtered.groupby('asin'):
        group = group.copy()
        group['block'] = group['price_changed'].cumsum()
        group['cleaned_price'] = group.groupby('block')['price'].transform('mean')
        cleaned_blocks.append(group[['asin', 'date', 'price_changed', 'cleaned_price']])

    cleaned_prices_df = pd.concat(cleaned_blocks)

    # Merge cleaned data back with full df
    df = df.merge(cleaned_prices_df, on=['asin', 'date'], how='left')
    df['price_changed'] = df['price_changed'].fillna(0).astype(int)
    df['cleaned_price'] = df.groupby('asin')['cleaned_price'].ffill()
    df['price'] = df['cleaned_price']

    df.drop(columns=['cleaned_price'], inplace=True)
    return df

az_sellout_daily_cleaned = price_changes(az_sellout_daily, 3,0.05)
az_prices = az_sellout_daily_cleaned[['sap_code', 'date', 'price']].drop_duplicates()
#az_sellout_daily_cleaned['fk_price'] = az_sellout_daily_cleaned['fk_price'].replace(0, np.nan) 
az_sellout_daily_cleaned['fk_price'] = az_sellout_daily_cleaned['fk_price'].fillna(az_sellout_daily_cleaned['price'])
az_sellout_daily_cleaned['fk_gap'] = az_sellout_daily_cleaned['fk_price'] - az_sellout_daily_cleaned['price']

# az_sellout_daily_cleaned['log_price'] = np.log(az_sellout_daily_cleaned['price'])
# az_sellout_daily_cleaned['log_sales'] = np.log1p(az_sellout_daily_cleaned['sales'])
def compute_ref_price(group, alpha=0.8):
    group = group.sort_values('date')
    ref_prices = []

    prev_ref_price = None
    prev_price = None

    for i, row in group.iterrows():
        price = row['price']

        if prev_ref_price is None:
            # First period: reference price is current price
            ref_price = price
        else:
            ref_price = alpha * prev_ref_price + (1 - alpha) * price

        ref_prices.append(ref_price)

        # Update previous values for next iteration
        prev_price = price
        prev_ref_price = ref_price

    group['ref_price'] = ref_prices
    return group

def compute_reference_price_alt(df, beta=0.05):
    """
    Compute exponentially smoothed reference price for each sap_code in a DataFrame.
    df must contain: 'sap_code', 'date', 'price'
    Returns: original df with new column 'ref_price'
    """
    df = df.copy()
    df['ref_price'] = np.nan

    # Sort for stability
    df = df.sort_values(['sap_code', 'date'])

    # Apply per sap_code group
    for sap, group in df.groupby('sap_code'):
        group = group.copy()
        group = group.sort_values('date')
        prices = group['price'].values
        T = len(prices)
        ref_prices = np.zeros(T)

        # Initial reference price
        ref_prices[0] = prices[0]

        # Compute reference price for t ≥ 1
        for t in range(1, T):
            time_factors = np.exp(beta * np.arange(t))  # e^{β(s - t0)}
            weighted_sum = np.sum(time_factors * prices[:t])
            ref_prices[t] = np.exp(-beta * t) * (prices[0] + beta * weighted_sum)

        df.loc[group.index, 'ref_price'] = ref_prices

    return df

def flag_large_price_changes(df, threshold=0.6):
    df = df.sort_values(['sap_code', 'date']).copy()

    # Compute % change in price by sap_code
    df['price_pct_change'] = df.groupby('sap_code')['price'].pct_change()

    # Flag rows where absolute % change exceeds threshold
    df['large_price_change'] = df['price_pct_change'].abs() > threshold
    return df


def filter_extreme_price_changes(df, threshold=0.6):
    df = df.sort_values(['sap_code', 'date']).copy()

    def process_group(group):
        group = group.sort_values('date').copy()
        group['prev_price'] = group['price'].shift(1)
        group['perc_change'] = (group['price'] - group['prev_price']).abs() / group['prev_price']

        # Keep first row (no prev_price) and those with % change ≤ threshold
        return group[(group['prev_price'].isna()) | (group['perc_change'] <= threshold)]

    df_filtered = df.groupby('sap_code', group_keys=False).apply(process_group)
    return df_filtered

def add_price_change_columns(df):
    df = df.sort_values(['asin', 'date']).copy()

    df['num_days'] = 0
    df['prev_price'] = np.nan

    def process_group(group):
        group = group.sort_values('date').copy()
        prev_price_list = []
        num_days_list = []

        last_price = None      # Most recently observed price
        prev_price = None      # Price before the most recent change
        days_since_change = 0

        for i, row in group.iterrows():
            current_price = row['price']

            if last_price is None:
                # First row — set prev_price to current_price
                prev_price = current_price
                days_since_change = 0
            elif row['price_changed'] == 1:
                # On a price change: update prev_price to last_price
                prev_price = last_price
                days_since_change = 0
            else:
                days_since_change += 1

            prev_price_list.append(prev_price)
            num_days_list.append(days_since_change)

            # Update last_price after processing row
            last_price = current_price

        group['prev_price'] = prev_price_list
        group['num_days'] = num_days_list
        return group

    df = df.groupby('asin', group_keys=False).apply(process_group)
    return df




az_sellout_daily_cleaned.columns
# Apply the function group-wise
az_sellout_daily_cleaned = flag_large_price_changes(az_sellout_daily_cleaned,0.6)
az_sellout_daily_cleaned = az_sellout_daily_cleaned.groupby('sap_code', group_keys=False).apply(compute_reference_price_alt,0.3)
az_sellout_daily_cleaned = add_price_change_columns(az_sellout_daily_cleaned)
az_sellout_daily_cleaned['RG'] = az_sellout_daily_cleaned['price'] - az_sellout_daily_cleaned['ref_price']
az_sellout_daily_cleaned['PRG'] = 10**(-5)*(
    (az_sellout_daily_cleaned['price'] - az_sellout_daily_cleaned['ref_price'])
    .clip(lower=0)
)
#az_sellout_daily_cleaned.dropna(subset='fk_price',inplace=True)

#az_sellout_daily_cleaned['log_PRG'] = np.log1p(az_sellout_daily_cleaned['PRG'])  # use log1p in case of 0 PRG
az_sellout_daily_cleaned['NRG'] = 10**(-5)*(
    (az_sellout_daily_cleaned['ref_price'] - az_sellout_daily_cleaned['price'])
    .clip(lower=0)
)
#az_sellout_daily_cleaned['log_NRG'] = np.log1p(az_sellout_daily_cleaned['NRG'])  # use log1p in case of 0 NRG
az_sellout_daily_cleaned.sort_values(['sap_code','date'],inplace=True)
az_sellout_daily_cleaned['inventory_lag1'] = az_sellout_daily_cleaned.groupby('sap_code')['sellable_qty'].shift(1)
az_sellout_daily_cleaned['quarter'] = az_sellout_daily_cleaned['date'].dt.to_period('Q').astype(str)
az_sellout_daily_cleaned['inventory_inv'] = az_sellout_daily_cleaned.groupby(['asin','year','quarter'])['inventory_lag1'].transform('mean') / (az_sellout_daily_cleaned['sellable_qty']+1)
az_sellout_daily_cleaned['log_sales'] = np.log1p(az_sellout_daily_cleaned['sales'])  # use log1p in case of 0 sales
az_sellout_daily_cleaned['log_price'] = np.log(az_sellout_daily_cleaned['price'])
az_sellout_daily_cleaned['log_sales_lag1'] = az_sellout_daily_cleaned.groupby('sap_code')['log_sales'].shift(1)
az_sellout_daily_cleaned['NG_fk'] = 10**(-5)*(np.where(az_sellout_daily_cleaned['fk_gap'] > 0, abs(az_sellout_daily_cleaned['fk_gap']), 0))
az_sellout_daily_cleaned['PG_fk'] = 10**(-5)*(np.where(az_sellout_daily_cleaned['fk_gap'] < 0, abs(az_sellout_daily_cleaned['fk_gap']), 0))
az_sellout_daily_cleaned.dropna(subset=['inventory_lag1'],inplace=True)

def compute_min_price(df):
    df = df.copy()

    # Group by PF
    pf_groups = df.groupby('PF')

    min_prices = []

    for idx, row in df.iterrows():
        current_pf = row['PF']
        current_sap = row['sap_code']

        # Get all rows in same PF, excluding current sap_code
        candidates = pf_groups.get_group(current_pf)
        other_prices = candidates[candidates['sap_code'] != current_sap]['price']

        if other_prices.empty:
            min_prices.append(np.nan)
        else:
            min_prices.append(other_prices.min())

    df['min_price'] = min_prices
    return df

az_sellout_daily_cleaned = compute_min_price(az_sellout_daily_cleaned)
az_sellout_daily_cleaned['PF_diff'] = az_sellout_daily_cleaned['price'] - az_sellout_daily_cleaned['min_price']
az_sellout_daily_cleaned['PF_diff'] = az_sellout_daily_cleaned['PF_diff'].fillna(0)
az_sellout_daily_cleaned['price_diff'] = az_sellout_daily_cleaned['price'] - az_sellout_daily_cleaned['prev_price']
az_sellout_daily_cleaned['decay'] = 1 - np.exp(-0.2 *az_sellout_daily_cleaned['num_days'])
az_sellout_daily_cleaned['change_decayed'] = az_sellout_daily_cleaned['log_price']* az_sellout_daily_cleaned['decay']
clusters = pd.read_excel('az_clusters.xlsx')
az_sellout_daily_cleaned = pd.merge(az_sellout_daily_cleaned,clusters,'left','sap_code')
# az_sellout_daily_cleaned['change_decayed'] = az_sellout_daily_cleaned['price_diff']/(az_sellout_daily_cleaned['num_days']+1)
az_sellout_daily_cleaned.isna().sum()
print(az_sellout_daily_cleaned.dtypes)
cols_to_standardize = ['price','decay','fk_gap','RG','sales','log_sales', 'log_price', 'PRG', 'NRG', 'inventory_inv','NG_fk', 'PG_fk','PF_diff','price_diff','change_decayed','log_sales_lag1']
for col in cols_to_standardize:
    az_sellout_daily_cleaned[col + '_std'] = (
        az_sellout_daily_cleaned
        .groupby('cluster')[col]
        .transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) != 0 else 0)
    )
az_sellout_daily_cleaned.to_csv("az sellout\\az_transformed.csv")
