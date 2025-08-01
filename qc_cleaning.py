# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:33:17 2025

@author: HP
"""
import pandas as pd
import matplotlib.pyplot as plt
import random 
import numpy as np

qc_sellout = pd.read_excel("C:\\Users\\HP\\Downloads\\QC's L6M Sellout Data.xlsx")

qc_sellout = qc_sellout[qc_sellout['Category']=='Audio']
qc_sellout.head()
qc_sellout.columns = ['Channel', 'Business Group', 'Customer Group', 'City', 'Month', 'date',
       'Category', 'Segment', 'Sub-Category', 'sap_code', 'price', 'sales',
       'GMV']
qc_sellout['log_price'] = np.log(qc_sellout['price'])
qc_sellout['log_sales'] = np.log1p(qc_sellout['sales'])
qc_sellout['Customer Group'].unique()
qc_sellout.drop('Month',axis=1,inplace=True)
qc_sellout['month'] = qc_sellout['date'].dt.month
cols_to_standardize = ['log_sales', 'log_price']
for col in cols_to_standardize:
    qc_sellout[col + '_std'] = (
        qc_sellout
        .groupby('sap_code')[col]
        .transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) != 0 else 0)
    )
clusters = pd.read_excel("clusters\\qc_clusters.xlsx")
qc_sellout = pd.merge(qc_sellout,clusters,"inner","sap_code")
qc_sellout.to_csv("transformed\\qc_transformed.csv")
qc_sellout
qc_month = qc_sellout.groupby(['Material Code','Month','Customer Group',
                               'Segment', 'Sub-Category'])\
    .agg(total_gmv=('GMV','sum'),
         total_units=('Units','sum')).reset_index()
qc_month['Month'] = pd.to_datetime(qc_month['Month'], format='%b %y')
qc_month = qc_month.sort_values(by=['Material Code','Month'])

def monthly_asp_calc(df,gmv_col,units_col,id_col):
    df['asp'] = df[gmv_col]/df[units_col]
    df['asp'] = df['asp'].replace(0, np.nan)
    df['asp'] = df['asp'].replace([np.inf, -np.inf], np.nan)
    df['asp'] = df.groupby([id_col])['asp'].ffill()
    df = df.dropna()
    df['asp'] = np.round(df['asp'])
    return df
def first_intro(df,id_col,date_col):
    
    
def drr(df,units_col,month_col):
    df['drr'] = df['units'] / df['year_month'].dt.daysinmonth
    
qc_month = monthly_asp_calc(qc_month, 'total_gmv', 'total_units', 'Material Code')

unique_material_codes = qc_sellout['Material Code'].unique()
random_material_codes = random.sample(list(unique_material_codes), 5)

# Plotting
for material_code in random_material_codes:
    plt.figure(figsize=(10, 6))
    
    # Filter data for the specific material code
    material_df = qc_sellout[qc_sellout['Material Code'] == material_code]
    
    # Get unique customer groups for the material code
    customer_groups = material_df['Customer Group'].unique()
    
    for customer_group in customer_groups:
        # Filter data for the specific customer group
        customer_group_df = material_df[material_df['Customer Group'] == customer_group]
        
        # Plot Date vs. ASP
        plt.plot(customer_group_df['Date'], customer_group_df['ASP'], label=customer_group)
        
    plt.xlabel('Date')
    plt.ylabel('ASP')
    plt.title(f'ASP over Time for Material Code: {material_code}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
