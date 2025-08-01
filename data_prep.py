# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 09:41:12 2025

@author: HP
"""

import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
from calendar import monthrange


def price_changes(df, code, k=3, m=0.05):
    """
    Identify significant price changes for each product based on deviation
    from the last key price. Cleaned price is set as the average of prices
    between consecutive change points.
    
    Parameters:
        df (DataFrame): Input dataframe
        code (str): Grouping column (e.g., 'product_id')
        k (float): Std multiplier threshold
        m (float): Absolute percent change threshold
    
    Returns:
        DataFrame: df with 'price_changed' column added and price replaced with 'cleaned_price'
    """
    df = df.copy()
    df = df.sort_values([code, 'date'])

    # Work only with valid prices
    filtered = df[df['price'] > 0].copy()
    filtered['price_changed'] = 0
    filtered['perc_change'] = np.nan
    filtered['last_key_price'] = np.nan

    for code_num, group in filtered.groupby(code):
        prices = group['price'].values
        perc_changes = []
        key_flags = []
        last_key_price = None

        temp_pct = pd.Series(prices).pct_change().dropna()
        std_estimate = temp_pct.std() if not temp_pct.empty else 0

        for i, price in enumerate(prices):
            if last_key_price is None:
                perc_changes.append(0)
                key_flags.append(1)
                last_key_price = price
            else:
                pct = (price - last_key_price) / (last_key_price + 1e-6)
                condition = abs(pct) > k * std_estimate or abs(pct) > m
                if condition:
                    key_flags.append(1)
                    last_key_price = price
                else:
                    key_flags.append(0)
                perc_changes.append(pct)

        filtered.loc[group.index, 'perc_change'] = perc_changes
        filtered.loc[group.index, 'price_changed'] = key_flags

    # Calculate cleaned price as average of prices within blocks
    cleaned_blocks = []
    for code_num, group in filtered.groupby(code):
        group = group.copy()
        group['block'] = group['price_changed'].cumsum()
        group['cleaned_price'] = group.groupby('block')['price'].transform('mean')
        cleaned_blocks.append(group[[code, 'date', 'price_changed', 'cleaned_price']])

    cleaned_prices_df = pd.concat(cleaned_blocks)

    # Merge cleaned data back with full df
    df = df.merge(cleaned_prices_df, on=[code, 'date'], how='left')
    df['price_changed'] = df['price_changed'].fillna(0).astype(int)
    df['cleaned_price'] = df.groupby(code)['cleaned_price'].ffill()
    df['price'] = df['cleaned_price']

    df.drop(columns=['cleaned_price'], inplace=True)
    return df



# Amazon

az_2024 = pd.read_excel("C:\\Users\\HP\\Downloads\\2024 Amazon Data.xlsx",sheet_name="Data")
az_2025 = pd.read_excel("C:\\Users\\HP\\Downloads\\Boat sellout report till 27th July.xlsx")
common_cols = az_2024.columns.intersection(az_2025.columns)
az_sellout = pd.concat([az_2024[common_cols], az_2025[common_cols]], ignore_index=True)
az_sellout.fillna(0, inplace=True)
az_sellout['net_ordered_units'] = az_sellout['net_ordered_units'].round(0).astype(int)
sap_codes  = pd.read_excel("C:\\Users\\HP\\Downloads\\ASIN vs PF.xlsx")
sap_codes = sap_codes[sap_codes['SAP code']!=0]
sap_codes = sap_codes[['SAP code','asin','PF','Category']]
az_sellout.drop_duplicates(inplace=True,subset=['asin', 'snapshot_day','net_ordered_units'])
cols = ['net_ordered_units', 'net_ordered_gms_amt','sellable_qty', 'sellable_value']
az_sellout[cols] = az_sellout[cols].fillna(0)
az_sellout[cols] = az_sellout[cols].replace(r'^\s*$', np.nan, regex=True)
az_sellout[cols] = az_sellout[cols].astype(float)
az_sellout['price'] = np.where(az_sellout['net_ordered_units']==0,0,az_sellout['net_ordered_gms_amt']/az_sellout['net_ordered_units']*1.18)
az_sellout['snapshot_day'] = pd.to_datetime(az_sellout['snapshot_day'])
az_sellout['year'] = az_sellout['snapshot_day'].dt.year
az_sellout['month'] = az_sellout['snapshot_day'].dt.month

az_sellout_daily = az_sellout.loc[:,['week', 'year','month', 'snapshot_day', 'asin',
                                     'further_classification', 'net_ordered_units',
                                     'sellable_qty','price']]
az_sellout_daily = pd.merge(az_sellout_daily,sap_codes,how='inner',on='asin')
az_sellout_daily.columns = ['week', 'year', 'month', 'date', 'asin',
       'further_classification', 'sales', 'sellable_qty','price','sap_code','PF','Category']
az_sellout_daily = az_sellout_daily.rename({'price_x':'price','price_y':'fk_price'},axis=1)
az_sellout_daily = az_sellout_daily.sort_values(['asin', 'date'])

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

az_sellout_daily_cleaned = price_changes(az_sellout_daily, 'sap_code',2,0.05)

az_sellout_daily_cleaned.sort_values(['sap_code','date'],inplace=True)
az_sellout_daily_cleaned['inventory_lag1'] = az_sellout_daily_cleaned.groupby('sap_code')['sellable_qty'].shift(1)
az_sellout_daily_cleaned['quarter'] = az_sellout_daily_cleaned['date'].dt.to_period('Q').astype(str)
az_sellout_daily_cleaned['inventory_inv'] = az_sellout_daily_cleaned.groupby(['asin','year','quarter'])['inventory_lag1'].transform('mean') / (az_sellout_daily_cleaned['sellable_qty']+1)
az_sellout_daily_cleaned['log_sales'] = np.log1p(az_sellout_daily_cleaned['sales'])  # use log1p in case of 0 sales
az_sellout_daily_cleaned['log_price'] = np.log(az_sellout_daily_cleaned['price'])
az_sellout_daily_cleaned['log_sales_lag1'] = az_sellout_daily_cleaned.groupby('sap_code')['log_sales'].shift(1)
az_sellout_daily_cleaned.dropna(subset=['inventory_lag1'],inplace=True)
az_sellout_daily_cleaned.to_csv("data\\transformed\\az_transformed.csv")


#Flipkart Daily


def process_fk_data(df: pd.DataFrame, month: str, year: int = 2025) -> pd.DataFrame:
    """
    Processes a raw DataFrame by cleaning columns, unpivoting daily metrics
    (units, price, support), and refining the data. It assumes the data for
    daily columns extends to the last day of the specified month.

    Args:
        df (pd.DataFrame): The input DataFrame assumed to have a wide format
                           with daily unit, price, and support columns.
        month (str): The three-letter abbreviation of the month (e.g., 'May', 'Jun', 'Jul')
                     that the data pertains to. This is used for column naming
                     conventions (e.g., '31-May.2') and date parsing.
        year (int): The year the data pertains to. Defaults to 2025.

    Returns:
        pd.DataFrame: A processed DataFrame with 'units', 'price', and 'support'
                      in a long format, and cleaned date and categorical columns.
    """

    # Normalize month input to a consistent format (e.g., 'May', 'Jun', 'July')
    # and get its number for calendar calculations.
    try:
        month_full_name = datetime.strptime(month, '%b').strftime('%B')
        month_number = datetime.strptime(month, '%b').month
        month_cap = month.capitalize()
    except ValueError:
        # If the input is already a full name like 'July'
        month_full_name = month.capitalize()
        month_number = datetime.strptime(month, '%B').month
        month_cap = datetime.strptime(month, '%B').strftime('%b').capitalize()

    # Get the last day of the month
    last_day_of_month = calendar.monthrange(year, month_number)[1]

    # Remove unnamed columns (only those starting with 'Unnamed')
    df = df.loc[:, ~df.columns.str.contains(r'^Unnamed', regex=True)]

    # Determine the cutoff column based on the last day of the month ---
    cutoff_col = f'{last_day_of_month}-{month.capitalize()}.2'

    cutoff_idx = df.columns.get_loc(cutoff_col)
    df = df.iloc[:, :cutoff_idx + 1]

    # Split the wide columns into 3 value sets assuming first 12 columns are ID columns.
    id_cols = df.columns[:12]
    num_days = last_day_of_month
    # Define the column ranges based on `num_days` and `id_cols` length
    start_daily_cols = len(id_cols)
    units_cols = df.columns[start_daily_cols : start_daily_cols + num_days]
    price_cols = df.columns[start_daily_cols + num_days : start_daily_cols + 2 * num_days]
    support_cols = df.columns[start_daily_cols + 2 * num_days : start_daily_cols + 3 * num_days]

    # Melt each set
    units_melted = df[id_cols.tolist() + units_cols.tolist()].melt(
        id_vars=id_cols, var_name='date', value_name='units')
    price_melted = df[id_cols.tolist() + price_cols.tolist()].melt(
        id_vars=id_cols, var_name='date', value_name='price')
    support_melted = df[id_cols.tolist() + support_cols.tolist()].melt(
        id_vars=id_cols, var_name='date', value_name='support')

    # Normalize date strings (e.g., '1-June.1' -> '1-June')
    units_melted['date'] = units_melted['date'].str.replace(r'\.\d+', '', regex=True)
    price_melted['date'] = price_melted['date'].str.replace(r'\.\d+', '', regex=True)
    support_melted['date'] = support_melted['date'].str.replace(r'\.\d+', '', regex=True)
    
    # Ensure the month name in the 'date' column is consistent for parsing
    units_melted['date'] = units_melted['date'].str.replace(month_full_name.capitalize(), month_cap, regex=False)
    price_melted['date'] = price_melted['date'].str.replace(month_full_name.capitalize(), month_cap, regex=False)
    support_melted['date'] = support_melted['date'].str.replace(month_full_name.capitalize(), month_cap, regex=False)

    # Merge all three melted frames on ID columns + date
    final_df = units_melted.merge(price_melted, on=id_cols.tolist() + ['date'])
    final_df = final_df.merge(support_melted, on=id_cols.tolist() + ['date'])

    # Convert 'date' to datetime using the input 'year' and month_cap
    final_df['date'] = pd.to_datetime(final_df['date'] + '-2025', format='%d-%b-%Y')

    # Drop rows with NaN in critical identifier columns
    final_df = final_df.dropna(subset=['SAP Code','BAU MOP'])

    # Format 'SAP Code' to integer string
    final_df['SAP Code'] = final_df['SAP Code'].apply(lambda x: '{:.0f}'.format(x) if pd.notnull(x) else x)

    # Filter out products with fewer than 10 units sold last month
    sap_totals = final_df.groupby('SAP Code')['units'].sum()
    valid_saps = sap_totals[sap_totals >= 10].index
    final_df = final_df[final_df['SAP Code'].isin(valid_saps)].reset_index(drop=True)

    # Correct inconsistent category names
    category_fix_map = {
        'AIrdopes': 'Airdopes',
        'Rockerz- IN Ear': 'Rockerz- In Ear'
    }
    final_df['Category'] = final_df['Category'].replace(category_fix_map)

    return final_df

fk_may = pd.read_csv("C:\\Users\\HP\\Downloads\\FK Sales - Scattered Price & NM View - May'25.csv",header=2)
fk_jun = pd.read_csv("C:\\Users\\HP\\Downloads\\FK Sales - Scattered Price & NM View - Jun'25.csv",header=3)
fk_jul = pd.read_csv("C:\\Users\\HP\\Downloads\\FK Sales - Scattered Price & NM View - Jul'25.csv",header=3)
fk_may_final = process_fk_data(fk_may, 'May')
fk_jun_final = process_fk_data(fk_jun, 'Jun')
fk_jul_final = process_fk_data(fk_jul,"July")
fk_may_final.rename(columns={'Row Labels':'product_id'},inplace=True)
fk_jun_final.rename(columns={'Row Labels':'product_id'},inplace=True)
fk_jul_final.rename(columns={'FSN':'product_id'},inplace=True)
fk_apr = pd.read_excel("C:\\Users\\HP\\Downloads\\April_Daily_Sales_Data_FK.xlsx")
fk_apr['date'] = pd.to_datetime(fk_apr['date'], format='%Y%m%d')
fk_apr = fk_apr.groupby(['product_id','date']).agg(sales=('units','sum'),sales_amt=('gmv','sum')).reset_index()
fk_apr['price'] = fk_apr['sales_amt']/fk_apr['sales']

fk_apr_final = price_changes(fk_apr, 'product_id',3,0.05)
fk_apr_final.drop(['sales_amt'],axis=1,inplace=True)
fk_may_onward = pd.concat([fk_may_final,fk_jun_final,fk_jul_final],ignore_index=True)
fk_may_onward = price_changes(fk_may_onward,'product_id',m=0.02)

fk_may_onward = fk_may_onward[['product_id','date','price','units','price_changed']].copy()
fk_may_onward.columns= ['product_id', 'date', 'price', 'sales','price_changed']
fk_codes = fk_jul_final[['SAP Code','Model','product_id']].drop_duplicates()
fk_codes.columns = ['sap_code','model','product_id']
fk_final = pd.concat([fk_apr_final,fk_may_onward])

fk_final = pd.merge(fk_final,fk_codes,on='product_id')
fk_final.columns = ['product_id', 'date', 'sales', 'price', 'price_changed', 'sap_code','model']
fk_final['month'] = fk_final['date'].dt.month
fk_final['log_sales'] = np.log1p(fk_final['sales'])
fk_final['log_price'] = np.log(fk_final['price'])
fk_final = fk_final.dropna(subset=['log_sales','price'])
fk_final.to_csv("data\\transformed\\fk_transformed.csv")


# Flipkart monthly

fk_all_months = pd.read_csv("C:\\Users\\HP\\Downloads\\Copy of FK Sales - Monthly Sales.csv",header=2)
fk_all_months.head()

# Get previous month for filtering
today = datetime.now()
previous_month_date = today + relativedelta(months=-1)
formatted_previous_month = previous_month_date.strftime('%b %y')

fk_all_months = fk_all_months.loc[:,'SAP Code':'June 22 Sales']

# Drop unnamed columns
fk_all_months= fk_all_months.drop(columns=[col for col in fk_all_months.columns if 'Unnamed:' in col])
fk_all_months= fk_all_months.drop(columns=['% Contribution','Total Units in FY 25','Total Units in FY 26'])
fk_all_months.columns = fk_all_months.columns.str.strip()
cleaned_columns = []

for col in fk_all_months.columns:
    if col.endswith(' Sales'):
        # Remove ' Sales' from the end
        new_col_name = col.replace(' Sales', '')
        cleaned_columns.append(new_col_name)
    else:
        # Keep the original name
        cleaned_columns.append(col)

fk_all_months.columns = cleaned_columns


# Standardize date format of columns names
date_pattern = re.compile(r'\b([A-Za-z]+)\s*(\d{2})\b')

standardized_columns = []
for col in fk_all_months.columns:
    match = date_pattern.search(col)
    if match:
        month_part_raw = match.group(1)
        year_two_digit = match.group(2)
        try:
            # Try parsing with full month name (%B), then fall back to abbreviated (%b)
            try:
                parsed_date = datetime.strptime(f"{month_part_raw} {year_two_digit}", '%B %y')
            except ValueError:
                parsed_date = datetime.strptime(f"{month_part_raw} {year_two_digit}", '%b %y')
            standardized_columns.append(parsed_date.strftime('%b %y'))
        except ValueError:
            # If date parsing still fails, keep the original column name
            standardized_columns.append(col)
    else:
        # If no date pattern found, keep the original column name
        standardized_columns.append(col)
fk_all_months.columns = standardized_columns
fk_all_months.head()

# Unpivot dataframe
id_vars = fk_all_months.columns[:4].tolist()

fk_sellout_month = pd.melt(fk_all_months,
                       id_vars=id_vars,
                       var_name='Month_Year',
                       value_name='Sales')
fk_sellout_month.head()
fk_sellout_month.columns = ['sap_code', 'category', 'model', 'item_name', 'month_year',
       'sales']
# Convert SAP Code from scientific formatting to string 
fk_sellout_month.dropna(inplace=True,subset=['sap_code'])
fk_sellout_month['sap_code'] = fk_sellout_month['sap_code'].astype(int).astype(str)
fk_sellout_month['sales'] = fk_sellout_month['sales'].fillna(0)

def parse_month_year(x):
    for fmt in ('%b-%y', '%b %y'):
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            continue
    return pd.NaT  # Return missing value if neither format works

fk_sellout_month['date'] = fk_sellout_month['month_year'].apply(parse_month_year)

fk_sellout_month = fk_sellout_month.groupby(['sap_code', 'category', 'model','month_year','date'])['sales'].max().reset_index()

duplicates = fk_sellout_month[fk_sellout_month.duplicated(subset=['sap_code', 'date'], keep=False)]['sap_code'].unique()
fk_sellout_month = fk_sellout_month[~(fk_sellout_month['sap_code'].isin(duplicates))]

fk_sellout_month['period'] = (fk_sellout_month['date'].rank(method='dense').astype(int))

# Ensure fk_sellout_month is sorted by sap_code and date
fk_sellout_month = fk_sellout_month.sort_values(['sap_code', 'date'])

# First and last non-zero sale date for each sap_code
nonzero_sales = fk_sellout_month[fk_sellout_month['sales'] > 0]

first_dates = nonzero_sales.groupby('sap_code')['date'].min()
last_dates = nonzero_sales.groupby('sap_code')['date'].max()

# Filter rows within that range
fk_sellout_month = fk_sellout_month[
    fk_sellout_month.apply(
        lambda row: (
            row['sap_code'] in first_dates and
            first_dates[row['sap_code']] <= row['date'] <= last_dates[row['sap_code']]
        ),
        axis=1
    )
].reset_index(drop=True)

fk_sellout_month.to_excel("data\\fk_month_filtered.xlsx")


# QC

codes = pd.read_excel("data\\DIYAnalysis-2025-7-29-6-4-34-.xlsx")
codes = codes[['Category', 'Sub Category', 'Product Family',
       'Product Full Description', 'SAP Product Code']]
codes.columns = ['category', 'sub_category', 'PF',
       'prod_desc', 'sap_code']

qc_sellout = pd.read_excel("C:\\Users\\HP\\Downloads\\QC's L6M Sellout Data.xlsx")

qc_sellout = qc_sellout[qc_sellout['Category']=='Audio']
qc_sellout.head()
qc_sellout.columns = ['Channel', 'Business Group', 'Customer_Group', 'City', 'Month', 'date',
       'Category', 'Segment', 'Sub-Category', 'sap_code', 'price', 'sales',
       'GMV']
qc_sellout['log_price'] = np.log(qc_sellout['price'])
qc_sellout['log_sales'] = np.log1p(qc_sellout['sales'])
qc_sellout.drop('Month',axis=1,inplace=True)
qc_sellout['month'] = qc_sellout['date'].dt.month
qc_sellout = qc_sellout.merge(codes,how='inner',on='sap_code')
qc_sellout.to_csv("data\\transformed\\qc_transformed.csv")

# MOP
mop_pa = pd.read_excel("C:\\Users\\HP\\Downloads\\PA MOP.xlsx")
mop_pa.columns = ['sap_code', 'prod_desc', 'PF', 'MOP_pa']
mop = mop_pa[['sap_code','MOP_pa']]
az_mop = pd.read_excel("C:\\Users\\HP\\Downloads\\2025 Amazon Data.xlsx")
az_mop = az_mop.sort_values(['asin', 'snapshot_day'])
az_mop = az_mop.groupby('asin').last().reset_index()[['asin','NLCs']]
az_mop['MOP_az'] = az_mop['NLCs']/0.82
az_mop = az_mop[['asin', 'MOP_az']]
sap_codes = pd.read_excel("C:\\Users\\HP\\Downloads\\ASIN vs PF.xlsx")
sap_codes = sap_codes[sap_codes['SAP code']!=0]
sap_codes = sap_codes[['SAP code','asin']]
az_mop = pd.merge(az_mop, sap_codes, on='asin', how='inner')
az_mop.drop(columns=['asin'], inplace=True,axis=1)
az_mop = az_mop[['SAP code', 'MOP_az']]
az_mop.columns = ['sap_code','MOP_az']
az_mop['sap_code'] = az_mop['sap_code'].astype(int).astype(str)
mop = mop_pa.merge(az_mop,how="outer",on="sap_code")
mop['MOP'] = mop['MOP_pa']
mop.loc[~(mop['MOP_pa'] > 0), 'MOP'] = mop['MOP_az']
fk_jul_final = fk_jul_final.sort_values(['SAP Code','date'])
fk_mop = fk_jul_final.groupby('SAP Code')[['Billing MOP','BAU MOP']].last().reset_index()
fk_mop.columns = ['sap_code', 'J', 'MOP_fk']
mop = mop.merge(fk_mop, how="outer",on="sap_code")
mop['J'] = mop['J'].fillna(0)
mop.loc[~(mop['MOP']>0),'MOP'] = mop['MOP_fk']
mop = mop.dropna(subset=['sap_code', 'MOP'])
mop = mop[mop['MOP']>0]
mop = mop[['sap_code','MOP','J']]
mop = mop[pd.to_numeric(mop['sap_code'], errors='coerce').notna()]
mop.loc[:, 'sap_code'] = mop['sap_code'].astype(int).astype(str)
mop = mop.groupby('sap_code')[['MOP','J']].max().reset_index()
mop.value_counts('sap_code').sort_values()
mop.to_excel("data\\mop.xlsx")