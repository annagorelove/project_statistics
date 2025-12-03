import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import dask.dataframe as dd
import pymannkendall as mk

# Загружаем данные
excel_file = 'combined_output.xlsx'
df_pandas = pd.read_excel(excel_file)

# Преобразуем DataFrame pandas в DataFrame Dask
df_dask = dd.from_pandas(df_pandas, npartitions=4)

#3. Гипотеза: Есть тенденция роста/спада продаж во времени
series = df_dask['Объем_дал']


# Берем случайные 40% данных для оценки тренда
sample = df_dask.sample(frac=0.4)  
sample_values = sample['Объем_дал'].compute() 

# Рассчитываем тест Манна–Кендалла 
result = mk.original_test(sample_values)

# Вывод статистики, p-value и значимости тренда
print(f"Статистика теста: {result.slope} ({'возрастающий' if result.slope > 0 else 'убывающий'})")
print(f"P-значение: {result.p}")
print(f"Наличие значимого тренда: {'Да' if result.p < 0.05 else 'Нет'}")

# Интерпретация результата
if result.h:
    print(f"Обнаружена {result.trend} тенденция с p-value = {result.p}")
else:
    print("Тенденция не обнаружена")


#4. Гипотеза: Есть линейная регрессия между продажами
df = pd.read_excel('combined_output.xlsx', parse_dates=['дата'])
df['дата'] = pd.to_datetime(df['дата'])  
df['month'] = df['дата'].dt.to_period('M') 

# Агрегация данных по месяцам по объему продаж
monthly_sales = df.groupby('month')['Объем_дал'].sum().reset_index()
monthly_sales['month'] = monthly_sales['month'].dt.to_timestamp() 

# Корреляционный анализ
print("Корреляционная матрица:")
print(monthly_sales.corr())

# Линейная регрессия: пример зависимости продаж от времени
monthly_sales['month_number'] = np.arange(len(monthly_sales))
X = monthly_sales[['month_number']]
y = monthly_sales['Объем_дал']
X = sm.add_constant(X)

# Строим модель линейной регрессии
model = sm.OLS(y, X).fit()

# Получаем сводку модели, включая коэффициенты и p-value
print(model.summary())
y_pred = model.predict(X)

# Визуализация
plt.figure(figsize=(10,5))
sns.lineplot(x='month', y='Объем_дал', data=monthly_sales, marker='o', label='Фактические продажи')
plt.plot(monthly_sales['month'], y_pred, color='red', linestyle='--', label='Регрессионная линия')
plt.title('Продажи по месяцам и линейная регрессия')
plt.xlabel('Месяц')
plt.ylabel('Суммарные продажи')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()