import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import *

# Чтение и группировка данных 
df = pd.read_excel('combined_output.xlsx')
grouped_df = df.groupby('Брэнд', as_index=False)['Объем_дал'].sum()

#1. Гипотеза: Объем продаж подчиняется нормальному распределению
# Смотрим на распределение продаж по всему датафрейму
sns.histplot(df['Объем_дал'], kde=True)
plt.title('Распределение объема продаж')
plt.show()

# Смотрим на распределение продаж по сгруппированному датафрейму
sns.histplot(grouped_df['Объем_дал'], kde=True)
plt.title('Распределение объема продаж')
plt.show()

# Анализируем основные характеристики для сгруппированного датафрейма
print('Выборочное среднее:', grouped_df['Объем_дал'].mean())
print('Выборочная медиана:', grouped_df['Объем_дал'].median())
print('Выборочная мода:', grouped_df['Объем_дал'].mode())
print('Выборочное среднеквадратическое отклонение:', grouped_df['Объем_дал'].std())
print('Выборочный коэффициент асиметрии:', grouped_df['Объем_дал'].skew())

# Так же проводим тест Шапиро-Уилка на нормальность данных
statistic, p_value = shapiro(grouped_df['Объем_дал']) 

# Вывод статистики и p-value
print(f"Статистика Шапиро-Уилка: {statistic:.4f}")
print(f"p-value: {p_value:.4f}")

# Интерпретация результата
alpha = 0.05 
if p_value > alpha:
    print("Данные не противоречат нормальному распределению (нормальность принимается).")
else:
    print("Данные существенно отличаются от нормального распределения (нормальность отклоняется).")

# Визуализация приведенния данных к более нормальному виду
qqplot(grouped_df['Объем_дал'], line='s')
plt.show()

grouped_df.hist(column=['Объем_дал'],bins=10)
plt.show()

log_transform = np.log(grouped_df['Объем_дал'])
log_transform.hist(bins=10)
plt.show()

sqrt_transform = np.sqrt(grouped_df['Объем_дал'])
sqrt_transform.hist(bins=10)
plt.show()

cbrt_transform = np.cbrt(grouped_df['Объем_дал'])
cbrt_transform.hist(bins=10)
plt.show()

print('Исходные:', grouped_df['Объем_дал'].skew())
print('log_transform:', log_transform.skew())
print('sqrt_transform:', sqrt_transform.skew())
print('cbrt_transform:', cbrt_transform.skew())

#2. Гипотеза: Средний объем продаж различается между брендами
# группы
brand = ['Брэнд_1', 'Брэнд_2', 'Брэнд_3', 'Брэнд_4', 'Брэнд_5', 'Брэнд_6', 'Брэнд_7', 'Брэнд_8']
filtered_df=[]

# столбец для группировки
group = 'Брэнд'

# анализируемый признак
attribute = 'Объем_дал'

fig, axs = plt.subplots(len(brand), sharex=True, figsize=(10,10))

for i, val in enumerate(brand):
    # создаем отфильтрованные датареймы и записываем в список filtered_df
    filtered_df.append(grouped_df[(grouped_df[group] == val)])

    # строим гистограмму для признака attribute
    ax = axs[i]
    ax.hist(filtered_df[i][attribute], color='blue')

    # отображаем на гистограмме среднее значение признака attribute
    ax.axvline (x=filtered_df[i][attribute].mean(), color='red', linestyle='--')  
    ax.set_title(f"{val}")

plt.ylabel(attribute)
plt.tight_layout()
plt.show()

# Расчет критерия Крусскала–Уоллиса
stat, p = kruskal(filtered_df[0][attribute], filtered_df[1][attribute], 
        filtered_df[2][attribute], filtered_df[3][attribute], filtered_df[4][attribute],
        filtered_df[5][attribute], filtered_df[6][attribute], filtered_df[7][attribute])
print(f"Статистика H: {stat}")
print(f"p-значение: {p}")

# Интерпретация результата
if p < 0.05:
    print("Есть статистически значимые различия между группами.")
else:
    print("Нет статистически значимых различий между группами.")
