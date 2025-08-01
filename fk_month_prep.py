import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re


fk_all_months = pd.read_csv("C:\\Users\\HP\\Downloads\\Copy of FK Sales - Monthly Sales.csv",header=2)
fk_all_months.head()

# Get previous month for filtering
today = datetime.now()
previous_month_date = today + relativedelta(months=-1)
formatted_previous_month = previous_month_date.strftime('%b %y')


# Filter 
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

fk_sales = pd.melt(fk_all_months,
                       id_vars=id_vars,
                       var_name='Month_Year',
                       value_name='Sales')
fk_sales.head()
fk_sales.columns = ['sap_code', 'category', 'model', 'item_name', 'month_year',
       'sales']
# Convert SAP Code from scientific formatting to string 
fk_sales['sap_code'] = fk_sales['sap_code'].astype(pd.Int64Dtype()).astype(str)
fk_sales.to_excel("C:\\Users\\HP\\Downloads\\fk_month_filtered.xlsx")

