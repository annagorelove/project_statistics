import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from scipy.stats import spearmanr

# Чтение и группировка данных 
df = pd.read_excel('combined_output.xlsx', parse_dates=['дата'])
df['дата'] = pd.to_datetime(df['дата'])  
grouped_df = df.groupby('Брэнд', as_index=False)['Объем_дал'].sum()

#5. Гипотеза: Объем продаж меняется в зависимости от месяца
# Преобразование данных в широкую таблицу
df_pivot = df.groupby(['дата', 'Брэнд']).sum().reset_index()
df_wide = df_pivot.pivot(index='дата', columns='Брэнд', values='Объем_дал')
df_complete = df_wide.fillna(0)

# Расчет критерия Фридмана
stat, p = friedmanchisquare(df_complete['Брэнд_1'], df_complete['Брэнд_2'], df_complete['Брэнд_3'],
                            df_complete['Брэнд_4'], df_complete['Брэнд_5'], df_complete['Брэнд_6'],
                            df_complete['Брэнд_7'], df_complete['Брэнд_8'])

# Вывод статистики и p-value
print(f"Статистика Фридмана = {stat:.4f}, p-значение = {p:.4f}")

# Интерпретация результата
if p < 0.05:
    print("Есть статистически значимые различия между продажами брендов")
else:
    print("Статистических значимых различий между продажами брендов нет")

# Смотри на более детальнную сводку между продажами по критерию Фридмана
p_values = sp.posthoc_nemenyi_friedman(df_complete)
print(p_values)

# Расчет декомбозиции временных рядов 
df_ = pd.DataFrame({'Date': df['дата'], 'Value': df['Объем_дал']})
df_.set_index('Date', inplace=True)

# Выполнение сезонной декомпозиции
decomposition = seasonal_decompose(df_['Value'], model='additive', period=12)

# Визуализация компонентов
decomposition.plot()
plt.show()

#6. Гипотеза: Объемы продаж двух брендов коррелируют между собой 
# Расчет коэффициента Спирмена для бренда 1 и 2 
corr_1, p_value_1 = spearmanr(df_complete['Брэнд_1'], df_complete['Брэнд_2'])
print(f"Коэффициент Спирмена для Brand 1 и 2: {corr_1:.3f}, p-value: {p_value_1:.3f}")

# Расчет коэффициента Спирмена для бренда 3 и 4
corr_2, p_value_2 = spearmanr(df_complete['Брэнд_3'], df_complete['Брэнд_4'])
print(f"Коэффициент Спирмена для Brand 3 и 4: {corr_2:.3f}, p-value: {p_value_2:.3f}")

# Расчет коэффициента Спирмена для бренда 5 и 6
corr_3, p_value_3 = spearmanr(df_complete['Брэнд_5'], df_complete['Брэнд_6'])
print(f"Коэффициент Спирмена для Brand 5 и 6: {corr_3:.3f}, p-value: {p_value_3:.3f}")

# Расчет коэффициента Спирмена для бренда 7 и 8
corr_4, p_value_4 = spearmanr(df_complete['Брэнд_7'], df_complete['Брэнд_8'])
print(f"Коэффициент Спирмена для Brand 7 и 8: {corr_4:.3f}, p-value: {p_value_4:.3f}")



