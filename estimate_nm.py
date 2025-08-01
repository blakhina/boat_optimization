import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# NM Forecast at price optimization recommendation
opt_df = pd.read_excel("data\\opt_df.xlsx")
mop = pd.read_excel("data\\mop.xlsx")
opt_result = pd.read_excel("result\\optimization_results.xlsx")
forecasted_sellin_fk = pd.read_excel("result\\forecasted_sellin_fk.xlsx")
forecasted_sellin_az = pd.read_excel("result\\forecasted_sellin_az.xlsx")
opt_result['sap_code'] = opt_result['sap_code'].astype(str) 
opt_df = opt_df[['sap_code','wma_asp','epsilon_az','epsilon_fk']]
opt_result = opt_result.merge(opt_df,how="left",on="sap_code")

# Get date one month ahead
today = datetime.today()
next_month_date = today + relativedelta(months=1)
next_month_name = next_month_date.strftime('%B').lower()
print(next_month_name)

opt_result = opt_result.merge(forecasted_sellin_fk[
    ['sap_code','sales_'+next_month_name,'sellin_'+next_month_name]],how="inner",on="sap_code")
opt_result = opt_result.merge(mop,how="inner",on="sap_code")
cogs = pd.read_excel("C:\\Users\\HP\\Downloads\\July COGS.xlsx")
cogs = cogs[['Material Code','July COGS']]
cogs.columns = ['sap_code','C']
cogs['sap_code'] = cogs['sap_code'].astype(str)
opt_result = opt_result.merge(cogs,how="inner",on="sap_code")
# Gross margin only
d = [6,10,8,6]
def s_expr(i, p, epsilon, d, asp,sellin):
    prod = sellin
    for k in range(1, i + 1):
        p_k = p[k]
        p_km1 = asp if k == 1 else p[k - 1]
        prod *= p_k / p_km1
    p_i = p[i]
    p_im1 = asp if i == 1 else p[i - 1]
    return prod * (p_i / p_im1) ** epsilon * d[i - 1]

def demand_fk_from_row(row, d_fk=[6,10,8,6]):
    p = {
        1: row['BAU MOP'],
        2: row['DOTD'],
        3: row['Event'],
        4: row['BAU MOP']
    }
    epsilon_fk = row['epsilon_fk']
    asp = row['wma_asp']
    sellin = row['sales_august']
    
    return 0.81*row['BAU MOP']*sum(s_expr(i, p, epsilon_fk, d_fk, asp, sellin/30) for i in [1, 2, 3, 4])

def fk_profit_value(prices, epsilon_fk, asp, MOP, J, C,sellin_august, sellout_august, M_fk=0.19,d_fk=[6,10,8,6]):
    # Compute demand
    demand_fk = sum(
        s_expr(i, prices, epsilon_fk, d_fk, asp, sellin_august/30)
        for i in [1, 2, 3, 4]
    )
    
    # Compute penalty
    penalty = sum(
        0.75 * s_expr(i, prices, epsilon_fk, d_fk, MOP, sellout_august/30) *
        (-0.66 * prices[i] - 0.06 * MOP + 0.81 * J)
        for i in [1, 2, 3, 4]
    )
    
    # Compute profit
    profit = (
        demand_fk * (1 - M_fk) * MOP
        - C * demand_fk
        - penalty
    )
    
    return profit

# Row-wise az_profit_rule
def az_profit_value(prices, epsilon_az, asp, MOP, C, sellin_august, sellout_august, M_az=0.18, d_az=[6, 10, 8, 6]):
    # Compute demand
    demand_az = sum(
        s_expr(i, prices, epsilon_az, d_az, asp, sellin_august / 30)
        for i in [1, 2, 3, 4]
    )

    # Compute penalty
    penalty = sum(
        s_expr(i, prices, epsilon_az, d_az, MOP, sellout_august / 30) *
        (1 - M_az) * (MOP - prices[i])
        for i in [1, 2, 3, 4]
    )

    # Compute profit
    profit = (
        demand_az * (1 - M_az) * MOP
        - C * demand_az
        - penalty
    )

    return profit


opt_result['demand_fk'] = opt_result.apply(demand_fk_from_row, axis=1)

opt_result['fk_profit'] = opt_result.apply(
    lambda row: fk_profit_value(
        prices = {
            1: row['BAU MOP'],
            2: row['DOTD'],
            3: row['Event'],
            4: row['BAU MOP']
        },
        epsilon_fk=row['epsilon_fk'],
        asp=row['wma_asp'],
        MOP=row['BAU MOP'],  # MOP assumed to be same as price_1
        J=row['J'],
        C=row['C'],
        sellout_august=row['sales_august'],
        sellin_august=row['sales_august']
    ),
    axis=1
)
opt_result = opt_result[opt_result['C']>0]
# opt_result = opt_result[opt_result['sales_august']>0]
opt_result['NM'] = opt_result['fk_profit']/opt_result['demand_fk']
opt_result = opt_result[['sap_code','product','BAU MOP','DOTD','Event','C','J','sales_august','demand_fk','fk_profit','NM']]
opt_result.columns = ['sap_code', 'product', 'BAU MOP', 'DOTD', 'Event', 'COGS', 'MOP', 'sellout','Revenue','Profit','NM']
opt_result.to_excel("Flipkart Net margin.xlsx",index=False)


opt_result = pd.read_excel("optimization_results.xlsx")
opt_result['sap_code'] = opt_result['sap_code'].astype(str)
asp = opt_df[['sap_code','wma_asp','epsilon_az','epsilon_fk']]
opt_result = opt_result.merge(asp,how="left",on="sap_code")
opt_result = opt_result.merge(forecasted_sellin_az[['sap_code','sales_august','sellin_august']],how="inner",on="sap_code")
opt_result = opt_result.merge(mop,how="inner",on="sap_code")
cogs = pd.read_excel("C:\\Users\\HP\\Downloads\\July COGS.xlsx")
cogs = cogs[['Material Code','July COGS']]
cogs.columns = ['sap_code','C']
cogs['sap_code'] = cogs['sap_code'].astype(str)
opt_result = opt_result.merge(cogs,how="inner",on="sap_code")

# Gross margin only
d = [6,10,8,6]


opt_result['demand_az'] = opt_result.apply(demand_from_row_az, axis=1)

opt_result['az_profit'] = opt_result.apply(
    lambda row: az_profit_value(
        prices = {
            1: row['BAU MOP'],
            2: row['DOTD'],
            3: row['Event'],
            4: row['BAU MOP']
        },
        epsilon_az=row['epsilon_az'],
        asp=row['wma_asp'],
        MOP=row['BAU MOP'],  # Assume MOP is same as price_1
        C=row['C'],
        sellin_august=row['sales_august'],
        sellout_august=row['sales_august']
    ),
    axis=1
)

opt_result = opt_result[opt_result['C']>0]
# opt_result = opt_result[opt_result['sales_august']>0]
opt_result['NM'] = opt_result['fk_profit']/opt_result['demand_fk']
opt_result = opt_result[['sap_code','product','BAU MOP','DOTD','Event','C','J','sales_august','demand_fk','fk_profit','NM']]
opt_result.columns = ['sap_code', 'product', 'BAU MOP', 'DOTD', 'Event', 'COGS', 'MOP', 'sellout','Revenue','Profit','NM']
opt_result = opt_result.merge(sap_codes,"inner","sap_code")
opt_result.to_excel("AZ Net margin.xlsx",index=False)