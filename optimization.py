# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:34:58 2025

@author: HP
"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from amplpy import AMPL
ampl = AMPL()
ampl.set_option('solver', 'couenne')
import os
import importlib
from datetime import datetime, timedelta

def import_module_if_excel_old(module_name, excel_path, days_threshold=7):
    """
    Imports a module only if the Excel file was modified more than `days_threshold` days ago.
    
    Parameters:
        module_name (str): The name of the module (without .py).
        excel_path (str): Path to the Excel file.
        days_threshold (int): Number of days to consider as the threshold (default: 7).
    
    Returns:
        module or None: The imported module if condition is met; otherwise None.
    """
    if not os.path.exists(excel_path):
        print(f"File does not exist: {excel_path}")
        return None

    modified_time = datetime.fromtimestamp(os.path.getmtime(excel_path))
    threshold_time = datetime.now() - timedelta(days=days_threshold)

    if modified_time < threshold_time:
        print(f"Importing {module_name} as {excel_path} is older than {days_threshold} days.")
        return importlib.import_module(module_name)
    else:
        print(f"Not importing {module_name}: {excel_path} was modified within the last {days_threshold} days.")
        return None

# mod = import_module_if_excel_old("elasticity_estimation", "data\\unstandardized_coefficients.xlsx")


# Step 1: Tag platform
az_transformed = pd.read_csv("data\\transformed\\az_transformed.csv")
fk_transformed = pd.read_csv("data\\transformed\\fk_transformed.csv")
# qc_transformed = pd.read_csv("transformed\\qc_transformed.csv",dtype={"City": str})
elasticities = pd.read_excel("data\\unstandardized_coefficients.xlsx")
elasticities = elasticities.pivot(index='sap_code', columns='platform', values='elasticity').reset_index()
elasticities.columns = ['sap_code', 'epsilon_az', 'epsilon_fk']
cogs = pd.read_excel("data\\July COGS.xlsx")
cogs = cogs[['Material Code','July COGS']]
cogs.columns = ['sap_code','C']
elasticities.sap_code = elasticities.sap_code.astype(str)
cogs.sap_code = cogs.sap_code.astype(str)
az_final_filtered = az_transformed[(az_transformed['date']>=fk_transformed['date'].min()) & 
                                   (az_transformed['date']<=fk_transformed['date'].max())]\
                                    [['sap_code', 'date', 'sales', 'price']]
fk_final_filtered = fk_transformed[
    (fk_transformed['date'] >= az_transformed['date'].min()) & 
    (fk_transformed['date'] <= az_transformed['date'].max())
][['sap_code', 'date', 'sales', 'price']]
# fk_final_filtered = pd.merge(fk_final_filtered,fk_jun_final_join,how="left",on="sap_code")
# fk_final_filtered.drop(['sap_code'],axis=1,inplace=True)
fk_final_filtered['platform'] = 'fk'
az_final_filtered['platform'] = 'az'
# az_sellout_filtered.drop(['asin'],axis=1,inplace=True)
az_valid = az_final_filtered.sap_code.dropna().unique().astype(str)
fk_valid = fk_final_filtered.sap_code.dropna().unique().astype(str)
both=np.intersect1d(az_valid,fk_valid)
# Step 2: Standardize column names
# Rename sales column if needed (assumes `sales` and `price` are present)
# Example:
# fk_final_subset2.rename(columns={'units_sold': 'sales'}, inplace=True)



# Step 3: Concatenate
combined = pd.concat([fk_final_filtered, az_final_filtered], ignore_index=True)


# Step 1: Ensure 'date' is datetime
combined['date'] = pd.to_datetime(combined['date'])

# Step 2: Add week and month columns
combined['week'] = combined['date'].dt.to_period('W').apply(lambda r: r.start_time)
combined['month'] = combined['date'].dt.to_period('M').dt.to_timestamp()

# Step 3: Compute weighted price for ASP
combined['price_weighted'] = combined['price'] * combined['sales']
combined['sap_code'] = combined['sap_code'].astype(str)

# Step 4: Weekly summary (per sap_code, per platform)
weekly_summary = (
    combined.groupby(['platform', 'week', 'sap_code'])
    .agg(
        total_sales=('sales', 'sum'),
        total_price_weighted=('price_weighted', 'sum')
    )
    .reset_index()
)

# Step 5: Calculate ASP per group
weekly_summary['ASP'] = weekly_summary['total_price_weighted'] / weekly_summary['total_sales']

# Step 6: Pivot to wide format
weekly_pivot = weekly_summary.pivot(index=['week', 'sap_code'], columns='platform', values=['total_sales', 'ASP'])
weekly_pivot.columns = [f"{stat}_{platform}" for stat, platform in weekly_pivot.columns]
weekly_pivot = weekly_pivot.reset_index()

# Step 7: Fill missing sales with 0 (product not sold in week on platform)
weekly_pivot[['total_sales_fk', 'total_sales_az']] = weekly_pivot[['total_sales_fk', 'total_sales_az']].fillna(0)

# Step 8: Filter out sap_codes with persistent 0 sales on one platform
nonzero_fk = weekly_pivot.groupby('sap_code')['total_sales_fk'].sum() > 0
nonzero_az = weekly_pivot.groupby('sap_code')['total_sales_az'].sum() > 0
valid_sap_codes = nonzero_fk | nonzero_az
weekly_pivot = weekly_pivot[weekly_pivot['sap_code'].isin(valid_sap_codes[valid_sap_codes].index)]

# Step 9: Recalculate total sales and ASP
weekly_pivot['total_sales_all'] = weekly_pivot['total_sales_fk'] + weekly_pivot['total_sales_az']
# weekly_pivot['asp_ratio_fk_to_az'] = weekly_pivot['ASP_fk'] / weekly_pivot['ASP_az']

# Step 10: Fill ASP with 0 for missing values (ASP=0 usually indicates no sales)
weekly_pivot[['ASP_fk', 'ASP_az']] = weekly_pivot[['ASP_fk', 'ASP_az']].fillna(0)

# Step 11: Compute blended ASP
weekly_pivot['ASP'] = (
    weekly_pivot['ASP_fk'] * weekly_pivot['total_sales_fk'] +
    weekly_pivot['ASP_az'] * weekly_pivot['total_sales_az']
) / weekly_pivot['total_sales_all'].replace(0, np.nan)
weekly_pivot.ASP.sort_values()
weekly_pivot.isna().sum()
# Step 12: Merge with elasticity
weekly_pivot = pd.merge(weekly_pivot, elasticities, on='sap_code', how='left')

# Step 13: If ASP for a platform is missing (i.e. 0), treat its elasticity as 0
weekly_pivot['epsilon_fk'] = np.where(weekly_pivot['ASP_fk']==0, 0, weekly_pivot['epsilon_fk'])
weekly_pivot['epsilon_az'] = np.where(weekly_pivot['ASP_az']==0, 0, weekly_pivot['epsilon_az'])

# Step 14: Adjusted total sales using elasticity
weekly_pivot['adj_total_sales_fk'] = np.where(
    weekly_pivot['ASP_fk'] == 0,
    0,
    weekly_pivot['total_sales_fk'] * (weekly_pivot['ASP'] / weekly_pivot['ASP_fk']) ** weekly_pivot['epsilon_fk']
)
weekly_pivot['adj_total_sales_az'] = np.where(
    weekly_pivot['ASP_az'] == 0,
    0,
    weekly_pivot['total_sales_az'] * (weekly_pivot['ASP'] / weekly_pivot['ASP_az']) ** weekly_pivot['epsilon_az']
)

# Step 15: Total adjusted sales and platform proportions
weekly_pivot['adj_total_sales'] = weekly_pivot['adj_total_sales_fk'] + weekly_pivot['adj_total_sales_az']
weekly_pivot['sales_prop_fk'] = weekly_pivot['adj_total_sales_fk'] / weekly_pivot['adj_total_sales'].replace(0, np.nan)
weekly_pivot['sales_prop_az'] = weekly_pivot['adj_total_sales_az'] / weekly_pivot['adj_total_sales'].replace(0, np.nan)
weekly_pivot[weekly_pivot['sales_prop_az'].isna()]

weekly_pivot = (
    weekly_pivot.sort_values(['sap_code', 'week'])
    .groupby('sap_code')
    .tail(3)
)
# Remove sap_codes where any of the last 3 weeks has 0 total sales
valid_sap_codes = (
    weekly_pivot.groupby('sap_code')['total_sales_all']
    .apply(lambda x: (x > 0).all())
)

weekly_pivot = weekly_pivot[weekly_pivot['sap_code'].isin(valid_sap_codes[valid_sap_codes].index)]

def compute_platform_weights(df, date_col='week'):
    """
    Compute platform weights and asp using a 3-week weighted moving average (WMA)
    """
    df = df.copy()
    df = df.sort_values(['sap_code', date_col])

    weights = np.array([0.25,0.35,0.4])

    def weighted_avg(arr):
        if len(arr) < 3:
            return np.nan
        return np.dot(arr[-3:], weights)

    # Compute normalized platform weights
    def compute_weights(group):
        fk_wma = weighted_avg(group['sales_prop_fk'].values)
        az_wma = weighted_avg(group['sales_prop_az'].values)
        total = fk_wma + az_wma
        if total == 0:
            return pd.Series({'weight_fk': 0, 'weight_az': 0})
        return pd.Series({
            'weight_fk': fk_wma / total,
            'weight_az': az_wma / total
        })

    platform_weights = (
        df.groupby('sap_code')
        .apply(compute_weights)
        .reset_index()
    )

    # Compute WMA of ASP
    wma_asp = (
        df.groupby('sap_code')
        .apply(lambda g: pd.Series({
            'wma_asp': weighted_avg(g['ASP'].values)
        }))
        .reset_index()
    )

    # Merge all
    result = pd.merge(platform_weights, wma_asp, on='sap_code', how='outer')
    return result


# Step 17: Run the weight computation
platform_weights = compute_platform_weights(weekly_pivot)
platform_weights.value_counts('sap_code')
# Output preview
print(platform_weights.head())

opt_df = pd.merge(platform_weights, elasticities, on='sap_code', how='inner')
opt_df.fillna(0, inplace=True)
opt_df = pd.merge(opt_df, cogs, on='sap_code', how='inner')
mop = pd.read_excel("data\\mop.xlsx")
mop['sap_code']=mop['sap_code'].astype(str)
opt_df = pd.merge(opt_df, mop, on='sap_code', how='inner')
opt_df[opt_df['MOP']<opt_df['wma_asp']]

def optimize_minlp_product_couenne(row, m, n, M_dict, K):
    MOP = row['MOP']
    C = row['C']
    J = row['J']
    ASP = row['wma_asp']
    epsilon_fk = row['epsilon_fk']
    epsilon_az = row['epsilon_az']
    weight_fk = row['weight_fk']
    weight_az = row['weight_az']
    M_fk = M_dict['fk']
    M_az = M_dict['az']

    d_fk = np.array([6, 10, 8, 6])
    d_az = np.array([6, 10, 8, 6])

    model = pyo.ConcreteModel()

    model.z2 = pyo.Var(within=pyo.Integers)
    model.b2 = pyo.Var(within=pyo.Binary)
    model.z3 = pyo.Var(within=pyo.Integers)
    model.b3 = pyo.Var(within=pyo.Binary)

    model.p = pyo.Var([1, 2, 3, 4], within=pyo.NonNegativeReals)
    model.p[1].fix(ASP)
    model.p[4].fix(MOP)

    p2_min = MOP * (1 - m[0])
    p2_max = MOP * (1 - n[0])
    p3_min = MOP * (1 - m[1])
    p3_max = MOP * (1 - n[1])

    z2_lower = int(np.ceil((p2_min - 99) / 100))
    z2_upper = int(np.floor((p2_max - 49) / 100))
    z3_lower = int(np.ceil((p3_min - 99) / 100))
    z3_upper = int(np.floor((p3_max - 49) / 100))

    model.z2.setlb(z2_lower)
    model.z2.setub(z2_upper)
    model.z3.setlb(z3_lower)
    model.z3.setub(z3_upper)

    model.price_expr_p2 = pyo.Constraint(expr=model.p[2] == model.z2 * 100 + model.b2 * 50 + 49)
    model.price_expr_p3 = pyo.Constraint(expr=model.p[3] == model.z3 * 100 + model.b3 * 50 + 49)

    model.p[2].setlb(z2_lower * 100 + 49)
    model.p[2].setub(z2_upper * 100 + 99)
    model.p[3].setlb(z3_lower * 100 + 49)
    model.p[3].setub(z3_upper * 100 + 99)

    model.p_ordering = pyo.Constraint(expr=model.p[3] <= model.p[2])
    model.p3_floor = pyo.Constraint(expr=model.p[3] >= 1.1 * C)

    model.min_dev_p2 = pyo.Constraint(expr=(MOP - model.p[2]) / MOP >= n[0])
    model.min_dev_p3 = pyo.Constraint(expr=(model.p[2] - model.p[3]) / model.p[2] >= n[1])

    def s_expr(i, p, epsilon, d, mop):
        prod = 1
        for k in range(1, i + 1):
            p_k = p[k]
            p_km1 = mop if k == 1 else p[k - 1]
            prod *= p_k / p_km1
        p_i = p[i]
        p_im1 = mop if i == 1 else p[i - 1]
        return prod * (p_i / p_im1) ** epsilon * d[i - 1]

    def demand_fk_rule(m):
        return sum(s_expr(i, m.p, epsilon_fk, d_fk, MOP) for i in [1, 2, 3, 4])

    def demand_az_rule(m):
        return sum(s_expr(i, m.p, epsilon_az, d_az, MOP) for i in [1, 2, 3, 4])

    model.D_fk = pyo.Expression(rule=demand_fk_rule)
    model.D_az = pyo.Expression(rule=demand_az_rule)

    def objective_rule(m):
        return MOP * (
            (1 - M_fk) * weight_fk * m.D_fk +
            (1 - M_az) * weight_az * m.D_az
        )

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    def az_profit_rule(m):
        penalty = sum(s_expr(i, m.p, epsilon_az, d_az, MOP) * (1 - M_az) * (MOP - m.p[i]) for i in [1, 2, 3, 4])
        return weight_az*(m.D_az * (1 - K) * (1 - M_az) * MOP/1.18 - C * m.D_az - penalty/1.18) >= 0

    def fk_profit_rule(m):
        penalty = sum(0.75 * s_expr(i, m.p, epsilon_fk, d_fk, MOP) * (-0.66 * m.p[i] - 0.06 * MOP + 0.81 * J) for i in [1, 2, 3, 4])
        return weight_fk*(m.D_fk * (1 - K) * (1 - M_fk) * MOP/1.18 - C * m.D_fk - penalty/1.18) >= 0

    model.fk_profit = pyo.Constraint(rule=fk_profit_rule)
    model.az_profit = pyo.Constraint(rule=az_profit_rule)
    solver = pyo.SolverFactory('couenne')
    try:
        results = solver.solve(model, tee=False)
    except Exception as e:
        print(f"Solver failed for {row['sap_code']}: {e}")
        return {'sap_code': row['sap_code'], 'revenue': np.nan}

    if (results.solver.status == pyo.SolverStatus.ok) and \
       (results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible]):
        return {
            'sap_code': row['sap_code'],
            'price_1': MOP,
            'price_2': pyo.value(model.p[2]),
            'price_3': pyo.value(model.p[3]),
            'price_4': MOP,
            'revenue': pyo.value(model.objective)
        }
    else:
        print(f"Optimization failed for {row['sap_code']}. Status: {results.solver.status}, Condition: {results.solver.termination_condition}")
        return {'sap_code': row['sap_code'], 'revenue': np.nan}

def optimize_multi_product_minlp(product_df, m, n, M, K_init, min_K=0.0, step=0.01, min_n=0, n_step=0.02):
    results = []

    for i, row in product_df.iterrows():
        print(f"\nOptimizing product {i+1}/{len(product_df)}: {row['sap_code']}")
        current_n = n.copy()
        found_solution = False

        while current_n[0] >= min_n and current_n[1] >= min_n:
            K = K_init
            res = optimize_minlp_product_couenne(row, np.array(m), np.array(current_n), M, K)

            # Reduce K if result is NaN
            while pd.isna(res['revenue']) and K > min_K:
                K = round(K - step, 4)
                print(f"  Retrying with K={K} and n={current_n} for {row['sap_code']}")
                res = optimize_minlp_product_couenne(row, np.array(m), np.array(current_n), M, K)

            if not pd.isna(res['revenue']):
                found_solution = True
                res['K_used'] = K
                res['n_used'] = current_n
                results.append(res)
                break  # Stop trying lower n if success

            # Lower n values and try again
            current_n = [round(current_n[0] - n_step, 4), round(current_n[1] - n_step, 4)]
            print(f"  Lowering n to {current_n} for {row['sap_code']}")

        if not found_solution:
            # Log final failed attempt
            res['K_used'] = K
            res['n_used'] = current_n
            results.append(res)

    return pd.DataFrame(results)


if __name__ == '__main__':
    results_df = optimize_multi_product_minlp(
        product_df=opt_df[opt_df['MOP']>0],
        m=[0.2, 0.4],
        n=[0.1, 0.1],
        M={'fk': 0.19, 'az': 0.18},
        K_init=0.23,
        min_K=0.13
    )

    print("\n--- MINLP Optimization Results ---")
    print(results_df)

cogs = pd.read_excel("data\\July COGS.xlsx")
cogs = cogs[['Material Code','Material Description']]
cogs.columns = ['sap_code','product']
cogs.sap_code = cogs.sap_code.astype(str)
opt_results=pd.merge(results_df, cogs, on='sap_code', how='left')
opt_results['n_used'].dropna().apply(tuple).value_counts()
opt_results.drop(columns=['revenue','price_4','n_used'], inplace=True, axis=1)
opt_results.rename(columns={
    'price_1': 'BAU MOP',
    'price_2': 'DOTD',
    'price_3': 'Event',
}, inplace=True)
opt_results.to_excel("result\\optimization_results.xlsx", index=False)
opt_results = pd.read_excel("result\\optimization_results.xlsx")
failed_sap = opt_results[opt_results['BAU MOP'].isna()]['sap_code'].astype(str)
failed_df = opt_df[opt_df['sap_code'].isin(failed_sap)]
failed_df

opt_df.to_excel("data\\opt_df.xlsx")
## Loop order K->n
def optimize_multi_product_minlp_alt(product_df, m, n, M, K_init, min_K=0.0, step=0.01, min_n=0, n_step=0.02):
    results = []

    for i, row in product_df.iterrows():
        print(f"\nOptimizing product {i+1}/{len(product_df)}: {row['sap_code']}")
        K = K_init
        found_solution = False

        while K >= min_K:
            current_n = n.copy()

            while current_n[0] >= min_n and current_n[1] >= min_n:
                print(f"  Trying K={K} and n={current_n} for {row['sap_code']}")
                res = optimize_minlp_product_couenne(row, np.array(m), np.array(current_n), M, K)

                if not pd.isna(res['revenue']):
                    found_solution = True
                    res['K_used'] = K
                    res['n_used'] = current_n
                    results.append(res)
                    break  # Stop trying lower n if success

                current_n = [round(current_n[0] - n_step, 4), round(current_n[1] - n_step, 4)]
                print(f"    Lowering n to {current_n} for {row['sap_code']}")

            if found_solution:
                break  # Stop trying lower K if success

            K = round(K - step, 4)
            print(f"  Lowering K to {K} for {row['sap_code']}")

        if not found_solution:
            # Log final failed attempt
            res['K_used'] = K
            res['n_used'] = current_n
            results.append(res)

    return pd.DataFrame(results)

