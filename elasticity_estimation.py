import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import data_prep


def standardize_columns_by_group(df, group_col='group'):
    """
    Standardizes all numeric columns within each group in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        group_col (str): Column to group by for standardization (default: 'group')

    Returns:
        pd.DataFrame: DataFrame with standardized columns added (suffixed with '_std')
    """
    df = df.copy()
    
    # Identify numeric columns, excluding the group column
    numeric_cols = df.select_dtypes(include='number').columns.difference([group_col])
    
    for col in numeric_cols:
        df[col + '_std'] = (
            df.groupby(group_col)[col]
              .transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) != 0 else 0)
        )
    
    return df


# Fit OLS with the same fixed effects

def fit_mixedlm(
    df,
    formula,
    re_formula,
    group_col='group',
    vc_formula=None,
):

    # Fit OLS model with same fixed effects formula
    
    # Filter data
    df = df.copy()
    
    # Fit Mixed Linear Model
    model = smf.mixedlm(
        formula=formula,
        data=df,
        groups=df[group_col],
        re_formula=re_formula,
        vc_formula=vc_formula
    )
    result = model.fit(reml=True, maxiter=1000)

    return result


# import data
group_col = ['PF','model','PF']
az_transformed = standardize_columns_by_group(pd.read_csv("data\\transformed\\az_transformed.csv"),group_col[0])
fk_transformed = standardize_columns_by_group(pd.read_csv("data\\transformed\\fk_transformed.csv"),group_col[1])
# qc_transformed = pd.read_csv("transformed\\qc_transformed.csv",dtype={"City": str})
# qc_transformed = qc_transformed.merge(codes,on='sap_code',how='inner')
# grouped_dfs = {}

# for group in qc_transformed['Customer_Group'].dropna().unique():
#     key = f"{group}".replace(" ", "_").replace("-", "_")
#     temp_df = qc_transformed[qc_transformed['Customer_Group'] == group].copy()
#     temp_df = standardize_columns_by_group(temp_df, group_col[2])
#     grouped_dfs[key] = temp_df

# zepto_transformed = grouped_dfs['Zepto']
# blinkit_transformed = grouped_dfs['Blinkit']
# swiggy_transformed = grouped_dfs['Swiggy']
# bbasket_transformed = grouped_dfs['Big_Basket']

formula = 'log_sales ~ log_price_std+C(month)'
formula_az = 'log_sales ~ log_price_std + inventory_inv_std + C(year)+C(month)'
re_formula = '~1+log_price_std'
vc_formula = None

# Amazon
result_az = fit_mixedlm(
    df=az_transformed,
    formula=formula_az,
    re_formula=re_formula,
    group_col=group_col[0],
    vc_formula=vc_formula
)

print(result_az.summary())

# Flipkart
result_fk = fit_mixedlm(
    df=fk_transformed,
    formula=formula,
    re_formula=re_formula,
    group_col=group_col[1],
    vc_formula=vc_formula
)

print(result_fk.summary())

# #QC
# result_qc = fit_mixedlm(
#     df=qc_transformed,
#     formula=formula,
#     re_formula=re_formula,
#     group_col='sap_code',
#     vc_formula=vc_formula
# )

# print(result_qc.summary())

def unstandardize_coefficients(result, df):
    betas = result.params.copy()
    unstd_betas = {}

    for param in betas.index:
        if param.endswith('_std'):
            orig_col = param.replace('_std', '')
            sigma_x = df[orig_col].std(ddof=0)
            unstd_betas[orig_col] = betas[param] / sigma_x
        else:
            unstd_betas[param] = betas[param]  # Categorical or unstandardized

    return pd.Series(unstd_betas)


def unstandardize_random_slopes_multi(result, df, random_cols, group_col='group',unstd_params=None):
    """
    Unstandardize multiple random slopes per group.

    Parameters:
    - result: fitted MixedLMResults object
    - df: DataFrame used to fit the model
    - random_cols: list of standardized column names used as random slopes
    - group_col: group name used for the random effects

    Returns:
    - DataFrame of unstandardized random effects per group
    """
    # Check only those columns are used that exist
    for col in random_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from df.")

    means = df[random_cols].mean()
    stds = df[random_cols].std()

    rows = []
    for group, re_dict in result.random_effects.items():
        row = {group_col: group}
        for col in random_cols:
            std = df[col.replace('_std', '')].std()
            if col in re_dict:
                row[col.replace('_std', '') + '_re_unstd'] = re_dict[col] * std
            else:
                row[col.replace('_std', '') + '_re_unstd'] = np.nan
            row[col.replace('_std', '') + '_coeff'] = re_dict[col] * std+unstd_params[col.replace('_std', '')]
        rows.append(row)

    return pd.DataFrame(rows)


unstd_params_az = unstandardize_coefficients(result_az, az_transformed)
unstd_params_fk = unstandardize_coefficients(result_fk, az_transformed)
# unstd_params_qc = unstandardize_coefficients(result_qc, az_transformed)

random_cols = ['log_price_std']
random_effects_az = unstandardize_random_slopes_multi(result_az, az_transformed, random_cols,group_col=group_col[0],unstd_params=unstd_params_az)
random_effects_fk = unstandardize_random_slopes_multi(result_fk, fk_transformed, random_cols,group_col=group_col[1],unstd_params=unstd_params_fk)
# random_effects_qc = unstandardize_random_slopes_multi(result_qc, qc_transformed, random_cols,group_col='sap_code',unstd_params=unstd_params_qc)

elasticities_az = pd.DataFrame(random_effects_az[[group_col[0],'log_price_coeff']])
elasticities_az = pd.merge(az_transformed[['sap_code',group_col[0]]].drop_duplicates(),elasticities_az, on=group_col[0], how='left')
elasticities_az.drop(columns=[group_col[0]], inplace=True)
elasticities_az['platform'] = 'az'
elasticities_fk = pd.DataFrame(random_effects_fk[[group_col[1],'log_price_coeff']])
elasticities_fk = pd.merge(fk_transformed[['sap_code',group_col[1]]].drop_duplicates(),elasticities_fk, on=group_col[1], how='left')
elasticities_fk.drop(columns=[group_col[1]], inplace=True)
elasticities_fk['platform'] = 'fk'
# elasticities_qc = pd.DataFrame(random_effects_qc[['sap_code','log_price_coeff']])
# elasticities_qc = pd.merge(qc_transformed['sap_code'].drop_duplicates(),elasticities_qc, on='sap_code', how='left')
# # elasticities_qc.drop(columns=['group'], inplace=True)
# elasticities_qc['platform'] = 'qc'
elasticities_df = pd.concat([elasticities_az, elasticities_fk], ignore_index=True)
elasticities_df.rename(columns={'log_price_coeff': 'elasticity'}, inplace=True)
elasticities_df.sort_values(by='elasticity')
elasticities_df.to_excel("data\\unstandardized_coefficients.xlsx", index=False)
