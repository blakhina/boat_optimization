import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from calendar import monthrange
from dateutil.relativedelta import relativedelta
import calendar
import importlib



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

mod = import_module_if_excel_old("elasticity_estimation", "unstandardized_coefficients.xlsx")


def compute_mixedlm_r2(df, result, target_col):

    # Fitted values and residuals
    df = df.copy()
    df['fitted'] = result.fittedvalues
    df['residual'] = df[target_col] - df['fitted']

    # Variance components
    var_resid = df['residual'].var()
    var_fixed = df['fitted'].var()
    var_random = float(result.cov_re.values.diagonal().sum())

    # R² calculations
    r2_marginal = var_fixed / (var_fixed + var_random + var_resid)
    r2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_resid)

    print(f"Marginal R² (fixed effects only): {r2_marginal:.4f}")
    print(f"Conditional R² (fixed + random effects): {r2_conditional:.4f}")

    return r2_marginal, r2_conditional


def create_future_data(df,platform ,forecast_horizon=5):
    """
    Create a DataFrame for future months for each SAP code with relevant modeling features.

    Parameters:
    - fk_sellout_month: DataFrame with columns ['sap_code', 'date', 'sales', 'period_sap', 'category']
    - forecast_horizon: int, number of future months to generate (default=5, e.g., Aug–Dec)

    Returns:
    - future_data: DataFrame with future months and features ['sap_code', 'date', 'month', 'year',
      'period_sap', 'sales_lag1', 'category', 'log_sales_lag1']
    """
    # Ensure datetime type
    df['date'] = pd.to_datetime(df['date'])

    # Step 0: Identify latest date
    max_date = df['date'].max()

    # Step 1: Create future months and sap_code grid
    future_months = pd.date_range(start=max_date + pd.offsets.MonthBegin(1), periods=forecast_horizon, freq='MS')
    sap_codes = df['sap_code'].unique()

    future_data = pd.MultiIndex.from_product(
        [sap_codes, future_months],
        names=['sap_code', 'date']
    ).to_frame(index=False)

    # Step 2: Add month and year
    future_data['month'] = future_data['date'].dt.month
    future_data['year'] = future_data['date'].dt.year
    future_data['month_sin'] = np.sin(2 * np.pi * future_data['month'] / 12)
    future_data['month_cos'] = np.cos(2 * np.pi * future_data['month'] / 12)
    # Step 3: Compute period_sap by continuing from last known period
    max_period_sap = df.groupby('sap_code')['period_sap'].max().fillna(0).astype(int)
    max_period = df.groupby('sap_code')['period'].max().fillna(0).astype(int)
    future_data = future_data.sort_values(['sap_code', 'date'])
    future_data['period_offset'] = future_data.groupby('sap_code').cumcount() + 1
    future_data['period'] = future_data['sap_code'].map(max_period) + future_data['period_offset']

    future_data['period_sap'] = (
        future_data.sort_values(['sap_code', 'date'])
        .groupby('sap_code').cumcount() + 1
    )
    future_data['period_sap'] += future_data['sap_code'].map(max_period_sap)
    future_data['festive'] = (
        ((future_data['year'] == 2024) & (future_data['month'].between(7, 10))) |
        ((future_data['year'] == 2023) & (future_data['month'] == 9)) |
        ((future_data['year'] == 2025) & (future_data['month'].between(8,11)))
    ).astype(int)
    future_data['festive_lag1'] = future_data.groupby('sap_code')['festive'].shift(1)
    future_data['festive_lag1'] = np.where((future_data['festive_lag1']==1) & 
                                                (future_data['festive']==1),1,0)
    future_data['festive_end'] = np.where((future_data['festive']==0) &
                                           (future_data['festive_lag1']==1),1,0)
    future_data = compute_festive_num(future_data)
    # Step 4: Add last known sales and category
    last_sales = df.sort_values('date').groupby('sap_code')['sales'].last()
    last_cat = df.sort_values('date').groupby('sap_code')['category'].last()
    future_data['sales_lag1'] = future_data['sap_code'].map(last_sales).astype(float)
    future_data['category'] = future_data['sap_code'].map(last_cat)
    
    # Step 5: Create log_sales_lag1
    future_data['log_sales_lag1'] = np.log1p(future_data['sales_lag1'])
    if platform=='amazon':
        last_price = df.sort_values('date').groupby('sap_code')['price'].last()
        future_data['log_price'] = np.log(future_data['sap_code'].map(last_price))
    return future_data

def predict_sellout(future_data, result, platform):
    """
    Predict monthly sellout from August to December using a mixed model.
    For each month, distribute total predicted sales proportionally based on
    product-level sales shares from the previous month.
    """
    future_data = future_data.copy()
    future_months = sorted(future_data['date'].unique())

    # Step 1: Predict raw sales using model
    raw_predictions = {}
    for current_month in future_months:
        curr_mask = future_data['date'] == current_month

        if platform == "amazon":
            X_pred_df = future_data.loc[curr_mask, [
                'sales_lag1', 'year', 'month', 'category', 'period_sap', 'log_price'
            ]].copy()
        else:
            X_pred_df = future_data.loc[curr_mask, [
                'sales_lag1', 'festive', 'year', 'festive_lag1', 'month_sin', 'month_cos'
            ]].copy()

        X_pred_df['sap_code'] = future_data.loc[curr_mask, 'sap_code'].values
        y_pred = result.predict(X_pred_df).clip(0)
        future_data.loc[curr_mask, 'predicted_sales_raw'] = y_pred
        raw_predictions[current_month] = y_pred.sum()

    # Step 2: Allocate predicted sales based on previous month's product share
    for i, current_month in enumerate(future_months):
        curr_mask = future_data['date'] == current_month

        # Total predicted sales in current month (from raw model)
        total_curr = future_data.loc[curr_mask, 'predicted_sales_raw'].sum()

        if i == 0:
            # For August: use actual lag sales from July
            prev_share = (
                future_data.loc[curr_mask, ['sap_code', 'sales_lag1']]
                .assign(share=lambda x: x['sales_lag1'] / x['sales_lag1'].sum())
                .set_index('sap_code')['share']
            )
        else:
            # From Sept onward: use predicted_sales from previous month
            prev_month = future_months[i - 1]
            prev_mask = future_data['date'] == prev_month

            prev_sales = future_data.loc[prev_mask, ['sap_code', 'predicted_sales']].copy()
            prev_sales = prev_sales[prev_sales['predicted_sales'] > 0]

            prev_share = (
                prev_sales.assign(share=lambda x: x['predicted_sales'] / x['predicted_sales'].sum())
                .set_index('sap_code')['share']
            )

        # Assign product-level predictions based on shares
        future_data.loc[curr_mask, 'predicted_sales'] = (
            future_data.loc[curr_mask, 'sap_code']
            .map(prev_share)
            .fillna(0) * total_curr
        )

        # Update lag for next month
        if i < len(future_months) - 1:
            next_month = future_months[i + 1]
            next_mask = future_data['date'] == next_month
            future_data.loc[next_mask, 'sales_lag1'] = future_data.loc[curr_mask, 'predicted_sales'].values
            future_data.loc[next_mask, 'log_sales_lag1'] = np.log1p(future_data.loc[next_mask, 'sales_lag1'])

    return future_data


def get_beginning_inventory(fk_inventory, fk_sellout_month, projected_sellin,sellout_date=pd.to_datetime('2025-07-25')):
    # Ensure datetime
    fk_inventory['stock_date'] = pd.to_datetime(fk_inventory['stock_date'])
    fk_sellout_month['date'] = pd.to_datetime(fk_sellout_month['date'])

    # Keep only July sellout
    fk_sellout_month = fk_sellout_month[
        fk_sellout_month['date'].dt.month == 7
    ]

    # Compute sellout from stock_date to July 31 for each sap_code
    def get_remainder_sellout(group):
        inventory_row = fk_inventory.loc[fk_inventory['sap_code'] == group.name, 'stock_date']

        if inventory_row.empty:
            return 0 

        stock_date = inventory_row.values[0]
        return group['sales'].sum() / sellout_date.day * ((sellout_date - stock_date).days)
    
    remainder_sellout = (
        fk_sellout_month
        .groupby('sap_code')
        .apply(get_remainder_sellout)
        .reset_index(name='sellout_july_remainder')
    )

    # Merge with fk_inventory and projected_sellin
    merged = (
        fk_inventory[['sap_code', 'stock']]
        .drop_duplicates('sap_code')
        .merge(remainder_sellout, on='sap_code', how='left')
        .merge(projected_sellin[['sap_code', 'sellin_july']], on='sap_code', how='left')
    )

    merged['sellout_july_remainder'] = merged['sellout_july_remainder'].fillna(0)
    merged['sellin_july'] = merged['sellin_july'].fillna(0)

    # Compute August beginning inventory
    merged['inventory_august'] = (merged['stock'] - merged['sellout_july_remainder'] + merged['sellin_july']).clip(0)

    return merged[['sap_code', 'inventory_august']]


def forecast_sellin_inventory(predicted_data, initial_inventory, start_month, start_year):
    """
    Calculate forecasted sell-in and ending inventory for 3 consecutive months.

    Inputs:
    - predicted_data: DataFrame with ['sap_code', 'month' (int), 'year' (int), 'predicted_sales']
    - initial_inventory: DataFrame with ['sap_code', 'inventory_<start_month_name>']
    - start_month: int (1-12) indicating the starting forecast month (e.g., 8 for August)
    - start_year: int indicating the starting year

    Output:
    - DataFrame with forecasted sell-in and inventory for 3 months
    """

    # Generate month-year labels for 5 months
    months = [(start_month + i - 1) % 12 + 1 for i in range(5)]
    years = [start_year + (start_month + i - 1) // 13 for i in range(5)]
    month_labels = [f"{calendar.month_name[m]} {y}" for m, y in zip(months, years)]
    sales_cols = [f"sales_{calendar.month_name[m].lower()}" for m in months]

    # Add month string to predicted_data
    predicted_data = predicted_data.copy()
    predicted_data['month_str'] = predicted_data.apply(
        lambda row: f"{calendar.month_name[row['month']]} {row['year']}", axis=1
    )

    # Pivot to wide format
    sales_wide = predicted_data.pivot(index='sap_code', columns='month_str', values='predicted_sales').reset_index()
    sales_wide.columns.name = None

    # Rename dynamically
    rename_map = {label: col for label, col in zip(month_labels, sales_cols)}
    sales_wide = sales_wide.rename(columns=rename_map)

    # Merge inventory
    start_month_name = calendar.month_name[start_month].lower()
    inv_col = f"inventory_{start_month_name}"
    df = sales_wide.merge(initial_inventory.rename(columns={inv_col: 'inventory_0'}), on='sap_code', how='left')

    # Convert sales & inventory to numeric
    for col in sales_cols:
        df[col] = pd.to_numeric(df.get(col), errors='coerce').fillna(0)
    df['inventory_0'] = pd.to_numeric(df['inventory_0'], errors='coerce').fillna(0)
    # df[sales_cols[0]] = df[['inventory_0', sales_cols[0]]].min(axis=1)
    # Forecast sell-in and inventory
    df['sellin_0'] = (df[sales_cols[1]] + 0.5*df[sales_cols[2]] - df['inventory_0']).clip(0)
    df['inventory_1'] = (df['inventory_0'] - df[sales_cols[0]] + df['sellin_0']).clip(0)

    df['sellin_1'] = (df[sales_cols[2]] + 0.5*df[sales_cols[3]] - df['inventory_1']).clip(0)
    df['inventory_2'] = (df['inventory_1'] - df[sales_cols[1]] + df['sellin_1']).clip(0)

    df['sellin_2'] = (df[sales_cols[3]] + 0.5*df[sales_cols[2]] - df['inventory_2']).clip(0)

    # Rename inventory/sell-in columns to month names
    inv_cols = [f"inventory_{calendar.month_name[m].lower()}" for m in months[:3]]
    sellin_cols = [f"sellin_{calendar.month_name[m].lower()}" for m in months[:3]]
    df.rename(columns={
        'inventory_0': inv_cols[0],
        'inventory_1': inv_cols[1],
        'inventory_2': inv_cols[2],
        'sellin_0': sellin_cols[0],
        'sellin_1': sellin_cols[1],
        'sellin_2': sellin_cols[2]
    }, inplace=True)

    return df[['sap_code'] + sellin_cols + sales_cols+inv_cols]


# Flipkart monthly sellout data
fk_sellout_month = pd.read_excel("data\\fk_month_filtered.xlsx",dtype={'sap_code':str})
projected_sellin = pd.read_excel("data\\24 july 2025 Stock Report with GIT (B2B+D2C).xlsx",sheet_name = "B2B Channel Wise",header=1,dtype={'sap_code':str})
projected_sellin = projected_sellin[['Material Code','Available Stock','Bucket Desc']]
projected_sellin.columns = ['sap_code','sellin_july','Bucket Desc']
projected_sellin['sap_code'] = projected_sellin['sap_code'].astype(str)
projected_sellin_fk = projected_sellin[projected_sellin['Bucket Desc']=='Flipkart'][['sap_code','sellin_july']]

fk_sellout_month['month_days'] = fk_sellout_month['date'].apply(lambda x: monthrange(x.year, x.month)[1])
fk_sellout_month['month'] = fk_sellout_month['date'].dt.month
fk_sellout_month['year'] = fk_sellout_month['date'].dt.year

fk_sellout_month['period_sap'] = (
    fk_sellout_month
    .assign(date=pd.to_datetime(dict(year=fk_sellout_month['year'], month=fk_sellout_month['month'], day=1)))
    .sort_values(['sap_code', 'date'])
    .groupby('sap_code')
    .cumcount() + 1
)

max_date = pd.to_datetime('2025-07-28')

# Daily Run Rate (drr) calculation
fk_sellout_month['drr'] = fk_sellout_month.apply(
    lambda row: row['sales'] / row['month_days'] if row['date'] < max_date
    else row['sales'] / (max_date.day if row['date'].month == max_date.month and row['date'].year == max_date.year else row['month_days']),
    axis=1
)
fk_sellout_month['log_sales'] = np.log1p(fk_sellout_month['sales']) 
fk_sellout_month['sales_lag1'] = fk_sellout_month.groupby('sap_code')['sales'].shift(1)
fk_sellout_month['drr_lag1'] = fk_sellout_month.groupby('sap_code')['drr'].shift(1)
fk_sellout_month['log_sales_lag1'] = fk_sellout_month.groupby('sap_code')['log_sales'].shift(1)

fk_inventory = pd.read_excel("data\\fk_inventory.xlsx") #Data manually copied from FK sales sheet

fk_inventory.dropna(inplace=True,subset=['SAP Code'])
fk_inventory = fk_inventory[['SAP Code','Stock','Date']]
fk_inventory.columns = ['sap_code', 'stock','stock_date']
fk_inventory['year'] = fk_inventory['stock_date'].dt.year
fk_inventory['month'] = fk_inventory['stock_date'].dt.month
fk_inventory['stock']=fk_inventory['stock'].fillna(0)
fk_inventory.loc[:,'sap_code'] = fk_inventory['sap_code'].astype(str).str.replace('.0', '', regex=False)

# Find sap's present in both sellout and inventory data in july
july_2025_sellout = fk_sellout_month[fk_sellout_month['date'].dt.to_period('M') == '2025-07']
july_2025_inventory = fk_inventory[fk_inventory['stock_date'].dt.to_period('M') == '2025-07']

# Get intersection of sap_codes
valid_sap_codes = set(july_2025_sellout['sap_code']).intersection(july_2025_inventory['sap_code'])

# Filter original datasets
fk_inventory = fk_inventory[fk_inventory['sap_code'].isin(valid_sap_codes)]
fk_sellout_month['festive'] = (
    ((fk_sellout_month['year'] == 2024) & (fk_sellout_month['month'].between(7, 10))) |
    ((fk_sellout_month['year'] == 2023) & (fk_sellout_month['month'] == 9))
).astype(int)
fk_sellout_month['festive_lag1'] = fk_sellout_month.groupby('sap_code')['festive'].shift(1)
# fk_sellout_month['festive_lag1'] = np.where((fk_sellout_month['festive_lag1']==1) & 
#                                             (fk_sellout_month['festive']==1),1,0)
def compute_festive_num(df):
    # Ensure proper sorting
    df = df.sort_values(['sap_code', 'date']).copy()

    # Step 1: Mark where a new festive period starts
    df['festive_shift'] = df.groupby('sap_code')['festive'].shift(fill_value=0)
    df['new_festive_block'] = (df['festive'] == 1) & (df['festive_shift'] == 0)

    # Step 2: Cumulative count of festive blocks
    df['festive_block_id'] = df.groupby('sap_code')['new_festive_block'].cumsum()

    # Step 3: Create festive_num: increasing within each festive block
    df['festive_num'] = df.groupby(['sap_code', 'festive_block_id']).cumcount() + 1

    # Set festive_num to 0 where festive is 0
    df.loc[df['festive'] == 0, 'festive_num'] = 0

    # Step 4: Apply quadratic transformation to festive_num
    # Values from 1 to 4 → peak at 2.5, scale to [0,1]
    df['festive_quad'] = - ((df['festive_num'] - 2.5) / 1.5) ** 2

    # Set to 0 where festive_num is 0
    df.loc[df['festive_num'] == 0, 'festive_quad'] = 0.0

    # Drop helper columns if not needed
    df = df.drop(columns=['festive_shift', 'new_festive_block', 'festive_block_id'])

    return df


fk_sellout_month = compute_festive_num(fk_sellout_month)
fk_sellout_month.dropna(subset='sales_lag1',inplace=True)
fk_sellout_month.isna().sum()
fk_sellout_month['category'] = fk_sellout_month['category'].str.strip().str.title()
fk_sellout_month['sales_lag1_sq'] = fk_sellout_month['sales_lag1']**2
fk_sellout_month['month_sin'] = np.sin(2 * np.pi * fk_sellout_month['month'] / 12)
fk_sellout_month['month_cos'] = np.cos(2 * np.pi * fk_sellout_month['month'] / 12)
fk_sellout_month['festive_end'] = np.where((fk_sellout_month['festive']==0) &
                                           (fk_sellout_month['festive_lag1']==1),1,0)
fk_sellout_month = fk_sellout_month[fk_sellout_month['year']>=2024]
fk_sellout_month = fk_sellout_month[fk_sellout_month['sap_code'].isin(valid_sap_codes)]
model = smf.mixedlm(
    formula='sales~sales_lag1+festive+festive:sales_lag1+month_sin+month_cos',
    data=fk_sellout_month,
    groups = fk_sellout_month['sap_code'],
    re_formula='~1'
)
# model = smf.ols('log_sales~1+log_sales_lag1+festive',data=fk_sellout_month)
result = model.fit(maxiter=10000)
print(result.summary())
r2_marginal, r2_conditional = compute_mixedlm_r2(fk_sellout_month, result,'sales')
# Step 2: Calculate Daily Run Rate (DRR)
fk_sellout_month['days_in_month'] = fk_sellout_month['date'].dt.daysinmonth
fk_sellout_month['drr'] = fk_sellout_month.apply(
    lambda row: row['sales'] / (row['days_in_month'] if row['date'] < max_date else max_date.day), axis=1
)
a= fk_sellout_month.groupby(['date'])['sales'].sum().reset_index()
# plt.plot(a['date'],a['sales'])
# plt.show()

# Step 3: Create future prediction months
future_months = pd.date_range(start=max_date + pd.offsets.MonthBegin(1), periods=4, freq='MS')
sap_codes = fk_sellout_month['sap_code'].unique()

# Step 0: Identify latest date in historical sellout

future_months = pd.date_range(start=max_date + pd.offsets.MonthBegin(1), periods=5, freq='MS')  # Aug–Dec

# Step 1: Create grid of future months x sap_codes

future_data = pd.MultiIndex.from_product(
    [valid_sap_codes, future_months],
    names=['sap_code', 'date']
).to_frame(index=False)

# Step 2: Add calendar and modeling features
future_data['month'] = future_data['date'].dt.month
future_data['year'] = future_data['date'].dt.year

# Step 3: Compute period_sap by continuing from last value
max_period = fk_sellout_month.groupby('sap_code')['period_sap'].max()
future_data['period_sap'] = (
    future_data.sort_values(['sap_code', 'date'])
    .groupby('sap_code').cumcount() + 1
)
future_data['period_sap'] += future_data['sap_code'].map(max_period).fillna(0).astype(int)

# Step 4: Add last known sales and category
last_sales = fk_sellout_month.sort_values('date').groupby('sap_code')['sales'].last()
last_cat = fk_sellout_month.sort_values('date').groupby('sap_code')['category'].last()

future_data['sales_lag1'] = future_data['sap_code'].map(last_sales)
future_data['category'] = future_data['sap_code'].map(last_cat)

future_data = create_future_data(fk_sellout_month,platform="flipkart",forecast_horizon=5)

predicted_data = predict_sellout(future_data, result,platform="flipkart")
predicted_data.dropna(subset='predicted_sales',inplace=True)
# predicted_data.to_excel("predicted_sales.xlsx")

projected_sellin_fk = projected_sellin[projected_sellin['Bucket Desc']=='Flipkart'].groupby('sap_code')['sellin_july'].sum().reset_index()
inventory_august = get_beginning_inventory(
    fk_inventory,
    fk_sellout_month,
    projected_sellin_fk
)

forecasted_sellin = forecast_sellin_inventory(predicted_data,inventory_august,start_month=8,start_year=2025)
forecasted_sellin.to_excel("result\\forecasted_sellin_fk_v8.xlsx")
forecasted_sellin = forecasted_sellin.merge(fk_sellout_month[['sap_code','model','item_name']].drop_duplicates(),how='inner',on='sap_code')


sap_codes  = pd.read_excel("C:\\Users\\HP\\Downloads\\ASIN vs PF.xlsx")
sap_codes = sap_codes[sap_codes['SAP code']!=0]
sap_codes = sap_codes[['SAP code','asin','item_name']]
sap_codes.columns = ['sap_code','asin','item_name']
sap_codes['sap_code'] = sap_codes['sap_code'].astype(str)
az_transformed = pd.read_csv("data\\transformed\\az_transformed.csv")
az_transformed['date'] = pd.to_datetime(az_transformed['date'])
az_transformed['day'] = az_transformed['date'].dt.day

az_sellout_month = (
    az_transformed
    .groupby(['Category','sap_code', 'year', 'month'])
    .agg(
        sales=('sales', 'sum'),
        drr = ('sales','mean'),
        price=('price', 'mean'),
        inventory=('sellable_qty', 'first'),
        start_date=('date', 'min'),
        date=('date', 'max')
    )
    .reset_index()
)
az_sellout_month['date'] = pd.to_datetime({
    'year': az_sellout_month['year'],
    'month': az_sellout_month['month'],
    'day': 1  # Use first of the month
})

b= az_sellout_month.groupby('date')['sales'].sum().reset_index()
plt.plot(b['date'],b['sales'])
plt.show()

az_sellout_month['log_sales'] = np.log1p(az_sellout_month['sales'])
az_sellout_month['log_price'] = np.log1p(az_sellout_month['price'])
az_sellout_month['sales_lag1'] = az_sellout_month.groupby('sap_code')['sales'].shift(1)
az_sellout_month['log_sales_lag1'] = az_sellout_month.groupby('sap_code')['log_sales'].shift(1)
# az_sellout_month['price_lag1'] = az_sellout_month.groupby('sap_code')['price'].shift(1)
# az_sellout_month['price_lag2'] = az_sellout_month.groupby('sap_code')['price'].shift(2)
# az_sellout_month['period_sin'] = np.sin(2 * np.pi * az_sellout_month['period'] / 5)
# az_sellout_month['period_cos'] = np.cos(2 * np.pi * az_sellout_month['period'] / 5)
az_sellout_month['month_sin'] = np.sin(2 * np.pi * az_sellout_month['month'] / 12)
az_sellout_month['month_cos'] = np.cos(2 * np.pi * az_sellout_month['month'] / 12)
az_sellout_month['inventory_inv'] = 1/(az_sellout_month['inventory']+1)
az_sellout_month['period'] = (az_sellout_month['date'].rank(method='dense').astype(int))
az_sellout_month['period_sap'] = (
    az_sellout_month
    .assign(date=pd.to_datetime(dict(year=az_sellout_month['year'], month=az_sellout_month['month'], day=1)))
    .sort_values(['sap_code', 'date'])
    .groupby('sap_code')
    .cumcount() + 1
)
az_sellout_month.dropna(inplace=True)
# sellout model
az_sellout_month.rename(columns={'Category':'category'},inplace=True)
model = smf.mixedlm('sales~sales_lag1+C(year)+C(category)+C(month)+period_sap+log_price+period_sap',
                    data=az_sellout_month,
                    groups=az_sellout_month['sap_code'],
                    re_formula='~1')
result_az = model.fit(maxiter=10000)
print(result_az.summary())
r2_marginal, r2_conditional = compute_mixedlm_r2(az_sellout_month, result_az,'log_sales')

future_data_az = create_future_data(az_sellout_month,platform="amazon")
predicted_data_az = predict_sellout(future_data_az,result_az,platform="amazon")

az_inventory = az_sellout_month.sort_values('date').groupby('sap_code').agg(
    stock=('inventory','last'),
    stock_date=('date','max')).reset_index()
az_inventory['sap_code'] = az_inventory['sap_code'].astype(str)
projected_sellin_az = projected_sellin[projected_sellin['Bucket Desc']=="Clicktech Retai"].groupby('sap_code')['sellin_july'].sum().reset_index()
projected_sellin_az['sap_code'] = projected_sellin_az['sap_code'].astype(str)
az_sellout_month['sap_code'] = az_sellout_month['sap_code'].astype(str)
predicted_data_az['sap_code'] = predicted_data_az['sap_code'].astype(str)
az_pred_inventory = get_beginning_inventory(az_inventory,az_sellout_month,projected_sellin_az,sellout_date=az_transformed.date.max())
# Predict sales
forecasted_sellin_az = forecast_sellin_inventory(predicted_data_az,az_pred_inventory,start_month=8,start_year=2025)
forecasted_sellin_az = pd.merge(forecasted_sellin_az,sap_codes,how="left",on="sap_code")
forecasted_sellin_az['sales_august'].sum()
forecasted_sellin_az.to_excel("result\\Amazon Sellin Forecast.xlsx")



# az_sellout_month['month_year'] = az_sellout_month['date'].dt.strftime('%b-%Y')

# # Pivot the table
# pivot = az_sellout_month.pivot_table(
#     index='sap_code',
#     columns='month_year',
#     values='sales',
#     aggfunc='sum',
#     fill_value=0
# )

# # Optional: sort columns by date
# pivot = pivot.reindex(sorted(pivot.columns, key=lambda x: pd.to_datetime(x, format='%b-%Y')), axis=1)
# predicted_data_az['month_year'] = predicted_data_az['date'].dt.strftime('%b-%Y')

# predicted_pivot = predicted_data_az.pivot_table(
#     index='sap_code',
#     columns='month_year',
#     values='predicted_sales',
#     aggfunc='sum',
#     fill_value=0
# )
# predicted_pivot = predicted_pivot.reindex(sorted(predicted_pivot.columns, key=lambda x: pd.to_datetime(x, format='%b-%Y')), axis=1)

# past_X = az_sellout_month[['sales_lag1', 'year', 'month', 'category', 'period_sap','log_price','sap_code']]
# past_X['predicted'] = result_az.predict(past_X)
# # past_X['predicted'] = np.expm1(past_X['predicted']).clip(0)
# # past_predicted_az = predict_sellout(az_sellout_month,result_az,platform="amazon")
# compare = pd.merge(past_X[['sap_code','year','month','predicted']],az_sellout_month[['sap_code','year','month','sales']],how='inner',on=['sap_code','year','month'])
# compare.to_excel("comparison_v2.xlsx")

# from sklearn.metrics import r2_score
# r2_score(compare['sales'],compare['predicted'])