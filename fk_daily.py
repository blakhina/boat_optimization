# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 10:41:43 2025

@author: HP
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import calendar
from datetime import datetime
import statsmodels.formula.api as smf

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
        # Use a full month name for parsing if the input is an abbreviation like 'Jun'
        # This handles cases where column names might use 'June' instead of 'Jun'
        month_full_name = datetime.strptime(month, '%b').strftime('%B') # e.g., 'Jun' -> 'June'
        month_number = datetime.strptime(month, '%b').month # e.g., 'Jun' -> 6
        month_cap = month.capitalize() # Use the original capitalization for column matching (e.g., 'Jun')
    except ValueError:
        # If the input is already a full name like 'July'
        month_full_name = month.capitalize()
        month_number = datetime.strptime(month, '%B').month
        month_cap = month.capitalize() # Use as is

    # Get the last day of the month
    last_day_of_month = calendar.monthrange(year, month_number)[1]

    # Step 1: Remove unnamed columns (only those starting with 'Unnamed')
    df = df.loc[:, ~df.columns.str.contains(r'^Unnamed', regex=True)]

    # --- Step 2: Determine the cutoff column based on the last day of the month ---
    # Construct the expected last support column name
    cutoff_col = f'{last_day_of_month}-{month_cap}.2'

    cutoff_idx = df.columns.get_loc(cutoff_col)
    df = df.iloc[:, :cutoff_idx + 1]

    # Step 3: Split the wide columns into 3 value sets
    # We still rely on the first 12 columns being ID columns. This is a critical assumption.
    id_cols = df.columns[:12]

    # The number of daily columns for units, price, and support is simply the last day of the month
    num_days = last_day_of_month
    
    # Define the column ranges based on `num_days` and `id_cols` length
    start_daily_cols = len(id_cols)
    units_cols = df.columns[start_daily_cols : start_daily_cols + num_days]
    price_cols = df.columns[start_daily_cols + num_days : start_daily_cols + 2 * num_days]
    support_cols = df.columns[start_daily_cols + 2 * num_days : start_daily_cols + 3 * num_days]


    if not all([len(units_cols) == num_days, len(price_cols) == num_days, len(support_cols) == num_days]):
        raise ValueError

    # Step 4: Melt each set
    units_melted = df[id_cols.tolist() + units_cols.tolist()].melt(
        id_vars=id_cols, var_name='date', value_name='units')
    price_melted = df[id_cols.tolist() + price_cols.tolist()].melt(
        id_vars=id_cols, var_name='date', value_name='price')
    support_melted = df[id_cols.tolist() + support_cols.tolist()].melt(
        id_vars=id_cols, var_name='date', value_name='support')

    # Step 5: Normalize date strings (e.g., '1-June.1' -> '1-June')
    # Replace any suffix like .1 or .2
    units_melted['date'] = units_melted['date'].str.replace(r'\.\d+', '', regex=True)
    price_melted['date'] = price_melted['date'].str.replace(r'\.\d+', '', regex=True)
    support_melted['date'] = support_melted['date'].str.replace(r'\.\d+', '', regex=True)
    
    # Ensure the month name in the 'date' column is consistent for parsing (e.g., 'Jun' or 'June')
    # If the original columns were '1-June', '2-June', etc., and input was 'Jun', this normalizes.
    # We use month_cap (e.g., 'Jun') for consistency with the format string.
    units_melted['date'] = units_melted['date'].str.replace(month_full_name.capitalize(), month_cap, regex=False)
    price_melted['date'] = price_melted['date'].str.replace(month_full_name.capitalize(), month_cap, regex=False)
    support_melted['date'] = support_melted['date'].str.replace(month_full_name.capitalize(), month_cap, regex=False)


    # Step 6: Merge all three melted frames on ID columns + date
    final_df = units_melted.merge(price_melted, on=id_cols.tolist() + ['date'])
    final_df = final_df.merge(support_melted, on=id_cols.tolist() + ['date'])

    # Convert 'date' to datetime using the input 'year' and canonical month_cap
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
fk_may_final = process_fk_data(fk_may, 'May')
fk_jun_final = process_fk_data(fk_jun, 'Jun')

fk_apr = pd.read_excel("C:\\Users\\HP\\Downloads\\April_Daily_Sales_Data_FK.xlsx")
fk_apr['date'] = pd.to_datetime(fk_apr['date'], format='%Y%m%d')
fk_apr = fk_apr.groupby(['product_id','date']).agg(sales=('units','sum'),sales_amt=('gmv','sum')).reset_index()
fk_apr['price'] = fk_apr['sales_amt']/fk_apr['sales']


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


fk_apr_final = price_changes(fk_apr, 'product_id')
fk_apr_final.drop(['sales_amt'],axis=1,inplace=True)
fk_may_final = price_changes(fk_may_final,'Row Labels',m=0.02)
fk_jun_final = price_changes(fk_jun_final,'Row Labels',m=0.02)
fk_final = pd.concat([fk_may_final,fk_jun_final],ignore_index=True)

fk_final_subset = fk_final[['Row Labels','date','price','units','price_changed']].copy()
fk_final_subset.columns= ['product_id', 'date', 'price', 'sales','price_changed']
fk_codes = fk_jun_final[['SAP Code','Model','Row Labels']].drop_duplicates()
fk_codes.columns = ['sap_code','model','product_id']
#fk_models = fk_final_subset[['product_id','Category','Model']].drop_duplicates()
# fk_apr_final = pd.merge(fk_apr_final,fk_models,how='inner',on='product_id')
#fk_apr_final =  pd.merge(fk_apr_final,fk_codes,how='inner',on='product_id')
fk_final_subset2 = pd.concat([fk_apr_final,fk_final_subset])

fk_final_jun_merge = fk_jun_final[['Row Labels','SAP Code','Model']].drop_duplicates()
fk_final_jun_merge.columns = ['product_id','sap_code','model']
fk_final_subset2 = pd.merge(fk_final_subset2,fk_final_jun_merge,on='product_id')
az_prices = pd.read_excel("az_prices.xlsx")
az_prices['sap_code'] = az_prices['sap_code'].astype(str)
fk_sellout_daily = pd.merge(fk_final_subset2, az_prices, how='left', on=['sap_code', 'date'])
fk_sellout_daily = fk_sellout_daily[fk_sellout_daily['date']<='2025-06-26']
fk_sellout_daily['price_y'] = fk_sellout_daily['price_y'].fillna(fk_sellout_daily['price_x'])
fk_sellout_daily.columns = ['product_id', 'date', 'sales', 'price', 'price_changed', 'sap_code',
       'model', 'price_az']
fk_sellout_daily['az_gap'] = fk_sellout_daily['price'] - fk_sellout_daily['price_az']
fk_sellout_daily['month'] = fk_sellout_daily['date'].dt.month
fk_sellout_daily['log_sales'] = np.log1p(fk_sellout_daily['sales'])
fk_sellout_daily['log_price'] = np.log(fk_sellout_daily['price'])

def compute_min_price(df,group_col):
    df = df.copy()

    # Group by PF
    pf_groups = df.groupby(group_col)

    min_prices = []

    for idx, row in df.iterrows():
        current_pf = row[group_col]
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

fk_sellout_daily = compute_min_price(fk_sellout_daily, 'model')
fk_sellout_daily['min_price'] = fk_sellout_daily['min_price'].fillna(fk_sellout_daily['price'])
fk_sellout_daily['model_diff'] = fk_sellout_daily['price'] - fk_sellout_daily['min_price']
fk_sellout_daily = fk_sellout_daily.dropna(subset=['log_sales'])

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
            ref_price = alpha * prev_ref_price + (1 - alpha) * prev_price

        ref_prices.append(ref_price)

        # Update previous values for next iteration
        prev_price = price
        prev_ref_price = ref_price

    group['ref_price'] = ref_prices
    return group



fk_sellout_daily = fk_sellout_daily.groupby('sap_code', group_keys=False).apply(compute_ref_price)
fk_sellout_daily['RG'] = fk_sellout_daily['price'] - fk_sellout_daily['ref_price']
cols_to_standardize = ['az_gap','log_sales', 'log_price','model_diff','RG','ref_price']
for col in cols_to_standardize:
    fk_sellout_daily[col + '_std'] = (
        fk_sellout_daily
        .groupby('sap_code')[col]
        .transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) != 0 else 0)
    )

clusters = pd.read_excel("clusters\\fk_clusters.xlsx")
clusters.sap_code = clusters.sap_code.astype(str)
fk_sellout_daily = pd.merge(fk_sellout_daily,clusters,'left','sap_code')

fk_sellout_daily.to_csv("fk sellout\\fk_transformed.csv")
fk_sellout_daily.isna().sum()
model = smf.mixedlm(
    formula='log_sales_std ~ az_gap_std+log_price_std+C(month)',
    data=fk_sellout_daily,
    groups=fk_sellout_daily['sap_code'],
    re_formula="~0+log_price_std"
)
result = model.fit()
print(result.summary())

sales_std = fk_sellout_daily['log_sales'].std(ddof=0)
log_price_std = fk_sellout_daily['log_price'].std(ddof=0)

# Multiply standardized coefficients by (σ_y / σ_x) to reverse
unstd_coeffs = []

for sap, rand_eff in result.random_effects.items():
    # Total effect = fixed + random
    total_log_price_std = result.fe_params['log_price_std'] + rand_eff.get('log_price_std', 0)
    #total_RG_std = result.fe_params['RG_std'] + rand_eff.get('RG_std', 0)

    # Reverse standardization
    log_price_unstd = total_log_price_std * (sales_std / log_price_std)
    #RG_unstd = total_RG_std * (sales_std / RG_std)

    unstd_coeffs.append({
        'sap_code': sap,
        'log_price_unstd': log_price_unstd
        #'RG_unstd': RG_unstd
    })

unstd_df = pd.DataFrame(unstd_coeffs)
unstd_df.sort_values(by='log_price_unstd')
plt.hist(unstd_df['log_price_unstd'], bins=30, edgecolor='black')
plt.show()
unstd_df.to_excel("unstandardized_coefficients_fk.xlsx", index=False)



def aggregate_price_periods(df,code, max_chunk_days=5):
    df = df.copy()
    df = df.sort_values([code, 'date'])
    df['sales']= df['sales'].fillna(0)
    # Step 1: Identify continuous price periods
    df['price_period'] = df.groupby(code)['price_changed'].cumsum()

    agg_df = (
        df.groupby([code,'price','price_period'])
          .agg(
              total_sales=('sales', 'sum'),
              DRR=('sales', 'mean'),
              num_days=('sales', 'count'),
              first_date=('date', 'min'),
              last_date=('date', 'max')
          )
          .reset_index()
    )
    
    return agg_df

fk_final_price = aggregate_price_periods(fk_final_subset2, 'product_id')
fk_final_price['month'] = fk_final_price['first_date'].dt.month

fk_final_price = pd.merge(fk_final_price,fk_final_jun_merge,on='product_id')
product_sales = fk_jun_final.groupby('SAP Code')['units'].sum()
valid_product_ids = product_sales[product_sales >= 30].index
fk_final_price = fk_final_price[fk_final_price['sap_code'].isin(valid_product_ids)].copy()

# fk_final_price = pd.merge(fk_final_price,fk_models,how='left',on='product_id')

# import re

# def extract_color(text):
#     if pd.isna(text):
#         return "None"
#     text = text.lower()
#     for color in color_list:
#         if re.search(rf'\b{color}\b', text):
#             return color.capitalize()  # Optional: capitalize for presentation
#     return "None"

# # Basic color vocabulary (extend as needed)
# color_list = [
#     "black", "white", "red", "blue", "green", "yellow", "orange", "pink", "purple",
#     "brown", "grey", "gray", "beige", "gold", "silver", "maroon", "navy", "olive",
#     "teal", "turquoise", "lime", "indigo", "violet", "peach", "coral", "mustard"
# ]
# fk_final_price['clean_color'] = fk_final_price['Color'].apply(extract_color)
# # Step 1: Count number of unique products per color
# color_counts = fk_final_price.groupby("clean_color")["product_id"].nunique()

# # Step 2: Identify colors with fewer than 5 products
# rare_colors = color_counts[color_counts < 5].index.tolist()

# # Step 3: Set clean_color to NaN if it's a rare color
# fk_final_price["clean_color"] = fk_final_price["clean_color"].apply(
#     lambda x: x if x not in rare_colors else "None"
# )





# Step 1: Preprocessing

fk_final_price['log_sales'] = np.log1p(fk_final_price['total_sales']) 
fk_final_price['log_sales'] = np.log1p(fk_final_price['DRR'])      # use log1p in case of 0 sales
     # use log1p in case of 0 sales
fk_final_price['log_price'] = np.log(fk_final_price['price'])
group_var = 'model'

# Step 2: Standardize log_price within model (lowest level)
fk_final_price['log_price_scaled'] = fk_final_price.groupby(group_var)['log_price'].transform(
    lambda x: (x - x.mean()) / x.std(ddof=0)
)

# Optional: drop rows with NaNs (e.g., due to std=0)
fk_final_price = fk_final_price.dropna(subset=['log_price_scaled', 'log_sales'])
md = smf.mixedlm(
    formula="log_sales ~ log_price_scaled",
    data=fk_final_price,
    groups=group_var,
    re_formula="~1+log_price_scaled",  # random slope for price
    vc_formula={"month":"0+C(month)"}
)

result = md.fit()
result.summary()
# STEP 3: Extract fixed and random effects
fixed = result.fe_params
global_intercept = fixed['Intercept']
global_slope = fixed['log_price_scaled']
random_effects = result.random_effects  # dict: {product_id: {'Intercept': ..., 'log_price_scaled': ...}}
group_counts = fk_final_price[group_var].value_counts().to_dict()

# STEP 4: Assemble product-specific coefficients
product_coeffs = []
for group, effects in random_effects.items():
    intercept = global_intercept
    slope_scaled = global_slope

    # Add random effects to intercept
    for key, value in effects.items():
        if key == 'log_price_scaled':
            slope_scaled += value
        else:
            intercept += value  
    n_obs = group_counts.get(group, 0)

    product_coeffs.append({
        group_var: group,
        'intercept': intercept,
        'slope_scaled': slope_scaled,
        'n_obs': n_obs
    })

product_df = pd.DataFrame(product_coeffs)

# STEP 5: Get per-product std of log_price to unscale the slope
log_price_std = fk_final_price.groupby(group_var)['log_price'].std(ddof=0).rename('log_price_std')
product_df = product_df.merge(log_price_std, on=group_var, how='left')

# STEP 6: Compute unscaled slope (original elasticity)
product_df['elasticity'] = product_df['slope_scaled'] / product_df['log_price_std']

# Final output
product_df = product_df[[group_var, 'elasticity','n_obs']]
product_df.columns = ['Model', 'elasticity','n_obs']
# Compare elasticities
el_compare = pd.merge(product_df,fk_elasticity,how="right",on="Model")
el_compare = el_compare[['FKN','sap_code', 'Model', 'elasticity_x', 'elasticity_y',
       'Billing MOP', 'Type', 'Price Range', 'Product Name', 'Color']]
el_compare.columns =['FKN','sap_code', 'Model', 'Elasticity (Model)', 'Elasticity (Product)',
       'Billing MOP', 'Type', 'Price Range', 'Product Name', 'Color']
el_compare.dropna(subset='Elasticity (Model)',inplace=True)
product_df = product_df[product_df['elasticity']<0]
from scipy.stats import chi2

# 2. Fit OLS for comparison
ols_model = smf.ols("log_sales ~ log_price_scaled+C(month)", data=fk_final_price).fit()

# 3. Log-likelihoods
print("MixedLM Log-likelihood:", result.llf)
print("OLS Log-likelihood:", ols_model.llf)

# 4. AIC/BIC manual calculation for MixedLM
n_params = len(result.fe_params) + result.cov_re.shape[0] + result.cov_re.shape[1] * (result.cov_re.shape[1] - 1) // 2
n_obs = fk_final_price.shape[0]
aic = -2 * result.llf + 2 * n_params
bic = -2 * result.llf + n_params * np.log(n_obs)
print(f"MixedLM AIC: {aic:.2f}")
print(f"MixedLM BIC: {bic:.2f}")

# 5. Likelihood Ratio Test
llr = 2 * (result.llf - ols_model.llf)
pval = chi2.sf(llr, df=n_params - len(ols_model.params))
print(f"Likelihood Ratio Test: χ² = {llr:.2f}, p = {pval:.4f}")

# 6. Approximate marginal and conditional R²
def calc_mixedlm_r2(result, df, y_col='log_sales'):
    y = df[y_col]
    y_hat_fixed = result.fittedvalues
    var_fixed = np.var(y_hat_fixed, ddof=1)
    var_total = np.var(y, ddof=1)

    # Sum all variances of random effects (diagonal of cov_re)
    var_random = result.cov_re.values.diagonal().sum()

    r2_marginal = var_fixed / var_total
    r2_conditional = (var_fixed + var_random) / var_total
    return r2_marginal, r2_conditional

r2_marginal, r2_conditional = calc_mixedlm_r2(result, fk_final_price)
print(f"Marginal R² (fixed effects only): {r2_marginal:.3f}")
print(f"Conditional R² (fixed + random): {r2_conditional:.3f}")

# 7. Residual plot
residuals = fk_final_price['log_sales'] - result.fittedvalues
plt.figure(figsize=(8, 4))
plt.scatter(result.fittedvalues, residuals, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.tight_layout()
plt.show()



from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

group_var = 'product_id'
# Step 2: Standardize log_price within model (lowest level)
fk_final_price['log_price_scaled'] = fk_final_price.groupby(group_var)['log_price'].transform(
    lambda x: (x - x.mean()) / x.std(ddof=0)
)
# Recompute global scaling if needed
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X = fk_final_price.reset_index(drop=True)
rmse_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    train_df = X.loc[train_idx].copy()
    test_df = X.loc[test_idx].copy()

    try:
        model = smf.mixedlm(
            formula="log_sales ~ log_price_scaled",
            data=fk_final_price,
            groups=fk_final_price[group_var],
            re_formula="~1+log_price_scaled",  # random slope for price
            vc_formula={"month":"0+C(month)"}
        )

        result = model.fit()

        y_pred = result.predict(test_df)
        rmse = root_mean_squared_error(test_df['log_sales'], y_pred)
        rmse_scores.append(rmse)

        print(f"Fold {fold + 1}: RMSE = {rmse:.4f}")
    except Exception as e:
        print(f"Fold {fold + 1} failed: {e}")

# Final average performance
print(f"\nAverage RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")



kf = KFold(n_splits=5, shuffle=True, random_state=42)
X = fk_final_price.reset_index(drop=True)

rmse_mixed_list = []
rmse_ols_list = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    train_df = X.loc[train_idx].copy()
    test_df = X.loc[test_idx].copy()

    # -----------------------------
    # Fit MixedLM on training data
    # -----------------------------
    try:
        model_mixed = smf.mixedlm(
            "log_sales ~ log_price_scaled",
            data=train_df,
            groups=train_df["product_id"],
            re_formula="~1 + log_price_scaled",  # random slope for price
            vc_formula={"month":"0+C(month)"}
        )
        result = model_mixed.fit()

        test_pred_mixed = result.predict(test_df)
        rmse_mixed = root_mean_squared_error(test_df['log_sales'], test_pred_mixed)
        rmse_mixed_list.append(rmse_mixed)
    except Exception as e:
        print(f"Fold {fold+1} MixedLM failed: {e}")
        rmse_mixed_list.append(np.nan)

    # -----------------------------
    # Fit OLS per product on train, predict on test
    # -----------------------------
    preds_ols = []
    actuals_ols = []

    for pid, group_test in test_df.groupby('product_id'):
        group_train = train_df[train_df['product_id'] == pid]

        if len(group_train) >= 5:  # minimum obs check
            lr = LinearRegression()
            X_train = group_train[['log_price_scaled']].values
            y_train = group_train['log_sales'].values
            X_test = group_test[['log_price_scaled']].values
            y_test = group_test['log_sales'].values

            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)

            preds_ols.extend(y_pred)
            actuals_ols.extend(y_test)

    if preds_ols:
        rmse_ols = root_mean_squared_error(actuals_ols, preds_ols)
        rmse_ols_list.append(rmse_ols)
    else:
        rmse_ols_list.append(np.nan)

    print(f"Fold {fold+1} -> RMSE MixedLM: {rmse_mixed:.4f}, RMSE OLS: {rmse_ols:.4f}")


agg = final_fk_may.groupby('Category').agg(Max=('price','max'),
                                           Min=('price','min'),
                                           Avg=('price','mean'),
                                           Count=('SAP Codes','count'))
agg['Range'] = agg['Max']-agg['Min']
fk_may_filtered = final_fk_may[['SAP Code','date','units','price']]

# remove all rows where the (SAP Code, date) appears more than once
num = fk_may_filtered[['SAP Code','date']].value_counts()
repeated_pairs = num[num > 1].index
fk_may_filtered_unique = fk_may_filtered[~fk_may_filtered.set_index(['SAP Code', 'date']).index.isin(repeated_pairs)].reset_index(drop=True)

filtered_df = fk_final_price[fk_final_price['product_id'].map(fk_final_price['product_id'].value_counts()) > 10]
# Ensure sorting by date if needed
filtered_df = filtered_df.sort_values(['product_id', 'price_period'])
product_sales = filtered_df.groupby('product_id')['total_sales'].sum()
valid_products = product_sales[product_sales >= 30].index
filtered_df = filtered_df[filtered_df['product_id'].isin(valid_products)]

# Get unique product IDs
product_ids = filtered_df['product_id'].unique()

# Plot each product
for pid in product_ids:
    subset = filtered_df[filtered_df['product_id'] == pid]

    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=subset, x='price', y='total_sales')
    plt.title(f'Sales vs Price for Product ID: {pid}')
    plt.xlabel('Price')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import OLSInfluence
from arch.bootstrap import MovingBlockBootstrap
import matplotlib.pyplot as plt

# Initialize results list
elasticity_results = []

for pid, group in filtered_df.groupby('product_id'):
    df = group.copy()
    df = df[(df['price'] > 0) & (df['DRR'] > 0)].sort_values('price_period')

    if len(df) < 10:
        continue
    
    df['log_price'] = np.log(df['price'])
    df['log_DRR'] = np.log(df['DRR'])
    
    try:
        # Fit base model and compute Cook's distance
        model = ols("log_DRR ~ log_price", data=df).fit()
        cooks_d = OLSInfluence(model).cooks_distance[0]
        threshold = 4 / len(df)
        df = df[cooks_d < threshold]

        # Refit model without outliers
        X = sm.add_constant(df['log_price'])
        y = df['log_DRR']
        final_model = sm.OLS(y, X).fit()
        base_elasticity = final_model.params['log_price']

        # Bootstrap for CI
        block_size = int(len(df) ** (1/3))
        bs = MovingBlockBootstrap(block_size, y, X)

        elasticities = []
        for data in bs.bootstrap(100):
            y_boot, X_boot = data[0], data[1]
            boot_model = sm.OLS(y_boot, X_boot).fit()
            elasticities.append(boot_model.params['log_price'])

        ci_low = np.percentile(elasticities, 2.5)
        ci_high = np.percentile(elasticities, 97.5)

        elasticity_results.append({
            'product_id': pid,
            'n_obs': len(df),
            'elasticity': base_elasticity,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'r_squared': final_model.rsquared
        })

        print(f"✅ {pid}: Elasticity = {base_elasticity:.2f}, CI = [{ci_low:.2f}, {ci_high:.2f}]")

    except Exception as e:
        print(f"❌ {pid}: failed with error {e}")

fk_may_agg = fk_may_filtered.groupby(['SAP Code','price']).agg(units = ('units','mean')).reset_index()

# Prepare a place to store results
demand_models = {}

# Loop through each product
for sap_code, group in fk_may_agg.groupby('SAP Code'):
    if group['price'].nunique() < 2:
        # Not enough variation to fit a model
        continue

    # Drop rows where units or price <= 0 to avoid log(0)
    group = group[(group['price'] > 0) & (group['units'] > 0)]
    if len(group) < 2:
        continue

    # Take logs
    X = np.log(group['price']).values.reshape(-1, 1)
    y = np.log(group['units']).values
    if X.shape[0] == 0 or y.shape[0] == 0:
        continue
    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Store results
    demand_models[sap_code] = {
        'model': model,
        'intercept': model.intercept_,
        'slope': model.coef_[0],  # This is the price elasticity
        'r_squared': model.score(X, y)
    }

positive_slope_saps = [sap for sap, v in demand_models.items() if v['slope'] > 0]
print("Number of SAP Codes with positive slope:", positive_slope_saps)

demand_models = {}
# Loop through each SAP Code
for sap_code, group in final_fk_may.groupby('SAP Code'):
    group = group.sort_values('date')

    # Calculate 3-day EMA for price and units
    group['price_ema'] = group['price'].ewm(span=3, adjust=False).mean()
    group['units_ema'] = group['units'].ewm(span=3, adjust=False).mean()

    # Drop rows with non-positive values (log undefined)
    group = group[(group['price_ema'] > 0) & (group['units_ema'] > 0)]
    if len(group) < 2:
        continue

    # Prepare log-log variables
    X = np.log(group['price_ema']).values.reshape(-1, 1)
    y = np.log(group['units_ema']).values

    if X.shape[0] == 0 or y.shape[0] == 0:
        continue

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Store results
    demand_models[sap_code] = {
        'model': model,
        'intercept': model.intercept_,
        'slope': model.coef_[0],  # elasticity
        'r_squared': model.score(X, y),
        'n_obs': len(group)
    }



# # Randomly sample 15 unique SAP Codes
# sampled_sap_codes = random.sample(fk_may_filtered['SAP Code'].unique().tolist(), 15)
# sampled_sap_codes = ['1000000218', '1000000441', '1000000447', '1000000475', '1000000549', '1000000945', '1000000946', '1000001099', '1000001130', '1000001186', '1000002287', '1000002456', '1000003012', '1000003013', '1000003163', '1000003185', '1000003186', '1000003222', '1000003306', '1000003320', '1000003321', '1000003363', '1000003426', '1000003445', '1000003571', '1000003623', '1000003629']
# # Filter data for sampled SAP Codes
# sampled_data = fk_may_filtered[fk_may_filtered['SAP Code'].isin(sampled_sap_codes)]

# # Plot for each SAP Code
# for sap in sampled_sap_codes:
#     product_data = sampled_data[sampled_data['SAP Code'] == sap].sort_values('date')

#     fig, ax1 = plt.subplots(figsize=(10, 4))

#     # Units (primary y-axis)
#     ax1.plot(product_data['date'], product_data['units'], color='tab:blue', label='Units Sold')
#     ax1.set_xlabel('Date')
#     ax1.set_ylabel('Units Sold', color='tab:blue')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')

#     # Price (secondary y-axis)
#     ax2 = ax1.twinx()
#     ax2.plot(product_data['date'], product_data['price'], color='tab:red', label='Price')
#     ax2.set_ylabel('Price', color='tab:red')
#     ax2.tick_params(axis='y', labelcolor='tab:red')

#     plt.title(f'SAP Code: {sap} - Units and Price over Time')
#     fig.tight_layout()
#     plt.show()
    
# # Now remove outliers- assume outliers are independent of price effect

fk_prices = pd.read_excel("C:\\Users\\HP\\Downloads\\DIYAnalysis-2025-7-9-10-44-18-.xlsx")
fk_prices = fk_prices.dropna(subset=['SAP Product Code'])
fk_prices = pd.melt(fk_prices, id_vars=['SAP Product Code'], var_name='date', value_name='price')
fk_prices['date'] = pd.to_datetime(fk_prices['date'].str.extract(r'(\d{2}-[A-Za-z]{3}-\d{4})')[0], format='%d-%b-%Y')
valid_sap_codes = fk_prices[fk_prices['price'].notna() & (fk_prices['price'] != 0)]['SAP Product Code'].unique()
fk_prices = fk_prices[fk_prices['SAP Product Code'].isin(valid_sap_codes)]
fk_prices = fk_prices.sort_values(['SAP Product Code', 'date'])

# Find the first valid price date for each SAP code
valid_dates = (
    fk_prices[fk_prices['price'].notna() & (fk_prices['price'] != 0)]
    .groupby('SAP Product Code')
    .agg(first=('date','min'),last=('date','max'))
    .reset_index()
)

# Merge this info back to the original dataframe
fk_prices = fk_prices.merge(valid_dates, on='SAP Product Code', how='left')

# Filter rows where date is on or after the first valid price date
fk_prices = fk_prices[fk_prices['date'] >= fk_prices['first']]
fk_prices = fk_prices[fk_prices['date'] <= fk_prices['last']]

# Drop helper column if no longer needed
fk_prices = fk_prices.drop(columns=['first','last'])
fk_prices['price'] = fk_prices['price'].fillna(method='ffill')  # Forward fill prices
fk_prices.to_csv("C:\\Users\\HP\\Downloads\\fk_prices.csv", index=False)