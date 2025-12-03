import pandas as pd
import numpy as np
file_path = 'sales_and_customer_base.xlsx'
sales_df = pd.read_excel(file_path, sheet_name='БД_объем')
client_df = pd.read_excel(file_path, sheet_name='справочник_клиент')
sku_df = pd.read_excel(file_path, sheet_name='справочник_СКЮ')
merged_df = pd.merge(sales_df, client_df, on='Код торговой точки', how='left')
df = pd.merge(merged_df, sku_df, on='SKU', how='left')
df.to_excel('combined_output.xlsx', index=False)
#df_nan = df.replace(0, np.nan)
# Удаляем строки с NaN
#df_no_zeros_dropna = df_nan.dropna()
#df_no_zeros_dropna.to_excel('combined_output.xlsx', index=False)
#df_no_zeros_dropna.info()
#print("Количество дубликатов:", df.duplicated().sum())

