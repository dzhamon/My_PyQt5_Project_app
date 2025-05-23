# Построение уравнения множественной регрессии и корреляционный анализ

import pandas as pd
import numpy as np
from utils.functions import CurrencyConverter
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def regresion_analyses(filtered_df, contract_df):
	
	# Получаем список номеров лотов, связанных с отфильтрованным проектом
	lot_numbers = filtered_df['lot_number'].unique()
	
	# Фильтруем контракты в таблице data_contract по номерам лотов
	filtered_contracts = contract_df[contract_df['lot_number'].isin(lot_numbers)]
	
	columns_info = [('total_contract_amount', 'contract_currency', 'total_contract_amount_eur')]
	
	# все суммы приводим к единой валюте EUR
	converter = CurrencyConverter()
	filtered_contracts = converter.convert_multiple_columns(filtered_contracts, columns_info=columns_info)
	
	# Преобразуем даты подписания контрактов в месячный формат
	filtered_contracts['contract_signing_month'] = filtered_contracts['contract_signing_date'].dt.to_period('M')
	
	# Убедимся, что нет пропущенных данных
	filtered_contracts = filtered_contracts.dropna(
		subset=['contract_signing_month', 'counterparty_name', 'total_contract_amount_eur'])
	
	# Выполняем группировку и разворот
	monthly_supplier_totals = filtered_contracts.groupby(['contract_signing_month', 'counterparty_name'])[
		'total_contract_amount_eur'].sum().unstack(fill_value=0)
	
	# Шаг 1: Определяем топ-20 поставщиков по сумме контрактов
	top_20_suppliers = filtered_contracts.groupby('counterparty_name')['total_contract_amount_eur'].sum().nlargest(
		20).index
	
	# Шаг 2: Создаем матрицу для регрессии, добавляем колонку для остальных поставщиков
	monthly_supplier_totals['Others'] = monthly_supplier_totals.drop(columns=top_20_suppliers, errors='ignore').sum(
		axis=1)
	top_20_matrix = monthly_supplier_totals[top_20_suppliers].copy()
	top_20_matrix['Others'] = monthly_supplier_totals['Others']
	
	# Добавляем зависимую переменную — общую сумму контрактов за каждый месяц
	top_20_matrix['Total_Contracts'] = top_20_matrix.sum(axis=1)
	
	# Шаг 1: Разделение данных на зависимые и независимые переменные
	X = top_20_matrix.drop(columns=['Total_Contracts'])
	y = top_20_matrix['Total_Contracts']
	
	# Шаг 2: Масштабирование данных (Ridge-регрессия чувствительна к масштабу переменных)
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	
	# Шаг 3: Разделение данных на обучающую и тестовую выборки
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
	
	# Шаг 4: Применение Ridge-регрессии
	ridge = Ridge(alpha=10.0)  # alpha — это параметр регуляризации (чем больше alpha, тем сильнее регуляризация)
	ridge.fit(X_train, y_train)
	
	# Шаг 5: Предсказание на тестовых данных
	y_pred = ridge.predict(X_test)
	
	# Шаг 6: Оценка модели
	print(f"R^2 на обучающих данных: {ridge.score(X_train, y_train)}")
	print(f"R^2 на тестовых данных: {ridge.score(X_test, y_test)}")
	print(f"Среднеквадратичная ошибка (MSE): {mean_squared_error(y_test, y_pred)}")
	
	# Получаем коэффициенты модели для каждого признака
	coef = ridge.coef_
	
	# Сопоставляем коэффициенты с именами признаков
	feature_importance = pd.Series(np.abs(coef), index=X.columns)
	
	# Сортируем по важности признаков
	feature_importance = feature_importance.sort_values(ascending=False)
	
	# Выводим важность признаков
	print(feature_importance)
	print('Коэффициенты регрессии')
	# Получаем и выводим коэффициенты регрессии (веса признаков)
	coefficients = pd.Series(ridge.coef_, index=X.columns)
	
	# Выводим коэффициенты
	print(coefficients)
	
	# Визуализация важности признаков
	plt.figure(figsize=(10, 6))
	feature_importance.plot(kind='bar')
	plt.title('Важность признаков (по абсолютным значениям коэффициентов)')
	plt.ylabel('Важность')
	plt.show()