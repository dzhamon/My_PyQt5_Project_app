import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QPushButton, QTableView, QAbstractItemView, QHBoxLayout,
                             QSizePolicy)
from utils.PandasModel_previous import PandasModel
import pandas as pd
from utils.functions import CurrencyConverter, save_analysis_results
from prophet import Prophet


def show_filtered_df(filtered_df):
	"""
	   Отображает отфильтрованные данные во всплывающем диалоговом окне.
	   """
	
	if filtered_df.empty:
		print('Нет данных для отображения!')
		return
	# Создание всплывающего диалогового окна
	dialog = QDialog()
	dialog.setWindowTitle("Отфильтрованные данные")
	dialog.setMinimumSize(800, 600)  # устанавливаем минм размер диалогового окна
	dialog.setMaximumSize(1900, 800)
	layout = QVBoxLayout(dialog)
	
	# Создаем виджет таблицы для отображения данных
	table_view = QTableView()
	model = PandasModel(filtered_df)
	table_view.setModel(model)
	table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
	layout.addWidget(table_view)
	
	# горизонтальная компоновка для кнопок
	button_layout = QHBoxLayout()
	
	# Кнопка для развертывания/сворачивания окна
	expand_button = QPushButton('Развернуть')
	expand_button.setCheckable(True)
	expand_button.toggled.connect(lambda checked: toggle_fullscreen(dialog, checked))
	button_layout.addWidget(expand_button)
	
	# кнопка для передачи данных на анализ
	analyze_button = QPushButton('Передать данные на анализ')
	analyze_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Фиксируем размер кнопки
	analyze_button.clicked.connect(lambda: send_data_to_analysis(filtered_df))
	button_layout.addWidget(analyze_button)
	
	# Добавляем кнопки в основной макет
	layout.addLayout(button_layout)
	dialog.setLayout(layout)
	dialog.exec_()


def send_data_to_analysis(filtered_df):
	"""
	Функция для передачи данных на анализ.
	"""
	print("Данные переданы на анализ:", filtered_df)


def toggle_fullscreen(dialog, checked):
	"""
	Разворачивает или возвращает окно к предыдущему размеру.
	"""
	if checked:
		dialog.showMaximized()
	else:
		dialog.showNormal()


def analyze_discrepancies(filtered_df):
	"""
	    Рассчитывает разницу и статус для каждого лота и возвращает результат в виде DataFrame.
	    """
	print('Мы в методе analyze_discrepancies')
	results = []
	print(filtered_df.columns)
	
	out_path = 'D:/My_PyQt5_Project_app/data/filtered_df.xlsx'
	filtered_df = filtered_df[
		filtered_df['total_price_kp'].notna() & filtered_df['total_contract_amount_contract'].notna()]
	filtered_df.to_excel(out_path, index=False)
	
	for _, row in filtered_df.iterrows():
		lot_number = row['lot_number']
		supplier_kp = row['winner_name']
		total_price_kp = row['total_price_kp']
		total_contract_amount_contract = row['total_contract_amount_contract']
		supplier_qty_kp = row['supplier_qty_kp']
		quantity_contract = row['quantity_contract']
		unit_price_kp = row['unit_price_kp']
		unit_price_contract = row['unit_price_contract']
		currency = row['currency']
		
		# Анализ изменений параметров
		if supplier_qty_kp == quantity_contract and unit_price_kp == unit_price_contract:
			status = 'Без изменений'
		elif (supplier_qty_kp * unit_price_kp) > (quantity_contract * unit_price_contract):
			status = 'Параметры занижены'
		elif (supplier_qty_kp * unit_price_kp) < (quantity_contract * unit_price_contract):
			status = 'Параметры контракта завышены '
		else:
			status = 'Следует проанализировать'
		
		# Добавляем все данные в список
		results.append({
			'lot_number': lot_number,
			'currency': currency,
			'winner_name': supplier_kp,
			'supplier_qty_kp': supplier_qty_kp,
			'unit_price_kp': unit_price_kp,
			'total_price_kp': total_price_kp,
			'quantity_contract': quantity_contract,
			'unit_price_contract': unit_price_contract,
			'total_contract_amount_contract': total_contract_amount_contract,
			'difference': total_contract_amount_contract - total_price_kp,
			'status': status
		})
	
	return pd.DataFrame(results)


def analyzeNonEquilSums(data_df, contract_df):
	# Агрегация данных по лотам в data_kp (для каждой позиции суммируем количество и цену)
	kp_agg = data_df.groupby(['lot_number', 'currency', 'winner_name']).agg(
		total_price_kp=pd.NamedAgg(column='total_price', aggfunc='sum'),
		supplier_qty_kp=pd.NamedAgg(column='supplier_qty', aggfunc='sum'),
		unit_price_kp=pd.NamedAgg(column='unit_price', aggfunc='sum'),
	).reset_index()
	#close_date_kp = pd.NamedAgg(column='close_date', aggfunc='max')
	
	# Агрегация данных по контрактам (по каждой позиции аналогично)
	contract_agg = contract_df.groupby(['lot_number', 'contract_currency', 'counterparty_name']).agg(
		total_contract_amount_contract=pd.NamedAgg(column='total_contract_amount', aggfunc='sum'),
		quantity_contract=pd.NamedAgg(column='quantity', aggfunc='sum'),
		unit_price_contract=pd.NamedAgg(column='unit_price', aggfunc='sum'),
	).reset_index()
	
	# contract_signing_date = pd.NamedAgg(column='contract_signing_date', aggfunc='max')
	
	merged_df = kp_agg.merge(
		contract_agg,
		left_on=['lot_number', 'currency', 'winner_name'],
		right_on=['lot_number', 'contract_currency', 'counterparty_name'],
		how='left',  # Используем left join, чтобы сохранить все строки из kp
		suffixes=('_kp', '_contract')
	)
	
	output_path = "D:/My_PyQt5_Project_app/data_analyze/merged_out.xlsx"
	merged_df.to_excel(output_path, index=False)
	
	# Фильтрация строк, где разница между суммой лота и контракта больше 0.01
	filtered_df = merged_df[abs(merged_df['total_price_kp'] - merged_df['total_contract_amount_contract']) > 0.01]
	
	analysis_results = analyze_discrepancies(filtered_df)
	
	save_analysis_results(analysis_results, output_path)
	
	# Вызов функции для отображения всплывающего окна
	show_filtered_df(analysis_results)


"""
	Тренд - анализ контрарактных данных
"""
def show_interactive_trend(filtered_df, output_folder):
	"""
   Показывает интерактивный график трендов стоимости контрактов для каждой дисциплины (топ-5 поставщиков).
   """
	columns_info = [('unit_price', 'contract_currency', 'unit_price_eur'),
	                ('total_contract_amount', 'contract_currency', 'total_contract_amount_eur')]
	converter = CurrencyConverter()
	filtered_df = converter.convert_multiple_columns(filtered_df, columns_info)
	
	# Убедимся, что папка для сохранения существует
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	# Получаем уникальные дисциплины
	disciplines = filtered_df['discipline'].unique()

	for discipline in disciplines:
		discipline_df = filtered_df[filtered_df['discipline'] == discipline].copy()
		discipline_df.loc[:, 'contract_signing_date'] = pd.to_datetime(discipline_df['contract_signing_date'])
		
		# Агрегируем данные для общего графика
		total_trend_data = discipline_df.resample('ME', on='contract_signing_date')['total_contract_amount_eur'].sum()
		
		# Сегментация по поставщикам
		supplier_totals = discipline_df.groupby('counterparty_name')['total_contract_amount_eur'].sum()
		top_suppliers = supplier_totals.nlargest(5).index  # Топ-5 поставщиков
		
		# Проверяем, есть ли хотя бы один поставщик
		if len(top_suppliers) > 0:
			# Создание интерактивного графика
			fig = make_subplots(rows=1, cols=1)
			
			# Добавляем общий график стоимости всех контрактов
			fig.add_trace(go.Scatter(x=total_trend_data.index, y=total_trend_data.values,
			                         mode='lines', name='Общая стоимость контрактов',
			                         line=dict(color='black', width=2)))
			
			# Добавляем графики для каждого поставщика
			for supplier in top_suppliers:
				supplier_df = discipline_df[discipline_df['counterparty_name'] == supplier]
				trend_data = supplier_df.resample('ME', on='contract_signing_date')['total_contract_amount_eur'].sum().fillna(0)
				fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data.values, mode='lines', name=supplier))
			
			# Настройка интерактивности
			fig.update_layout(title=f'Тренд стоимости контрактов для дисциплины: {discipline}',
			                  xaxis_title='Дата (год-месяц)', yaxis_title='Общая стоимость контрактов (EUR)',
			                  hovermode='x unified')
			
			# Сохранение графика в HTML
			output_file = os.path.join(output_folder, f"{discipline}_trend_analysis.html")
			fig.write_html(output_file)
		else:
			print(f"Недостаточно данных для анализа топ-5 поставщиков для дисциплины: {discipline}")
			
"""
	Моделирование и прогнозирование
"""
def prophet_and_arima(filtered_df):
	print('Мы в методе Prophet_and_Arima')
	print(filtered_df.columns)
	filtered_df = convert_to_eur(filtered_df) # приводим все суммы в единую валюту EUR
	
	# Переименуем столбцы для работы с Prophet
	df_prophet = filtered_df[['contract_signing_date', 'total_contract_amount_eur']].rename(
		columns={'contract_signing_date': 'ds', 'total_contract_amount_eur': 'y'})
	# Инициализируем модель Prophet
	model = Prophet()
	# Обучим модель на наших данных
	model.fit(df_prophet)
	
	# Прогнозируем на 12 месяцев вперед
	future = model.make_future_dataframe(periods=6, freq='ME')
	forecast = model.predict(future)
	
	# Выведем прогноз
	print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
	
	# Построим график прогноза
	model.plot(forecast)


def prepare_contract_data(filtered_df):
	# Функция для выполнения DBSCAN
	def perform_dbscan(data, eps=0.1, min_samples=3):
		dbscan = DBSCAN(eps=eps, min_samples=min_samples)
		data['cluster'] = dbscan.fit_predict(data)
		return data
	
	# Функция для построения k-NN графика
	def plot_knn_graph(data, k=5):
		nbrs = NearestNeighbors(n_neighbors=k).fit(data)
		distances, indices = nbrs.kneighbors(data)
		distances = np.sort(distances[:, k - 1])  # Сортировка по k-му соседу
		plt.figure(figsize=(10, 6))
		plt.plot(distances)
		plt.title(f"k-NN Graph for DBSCAN (k={k})")
		plt.xlabel("Points sorted by distance to k-th nearest neighbor")
		plt.ylabel("Distance")
		plt.grid()
		plt.show()
		return
	
	# Преобразование 'кг' в 'тонны'
	filtered_df.loc[filtered_df['unit'] == 'кг', 'quantity'] /= 1000
	filtered_df.loc[filtered_df['unit'] == 'кг', 'unit'] = 'тонны'
	
	# Нормализация данных
	scaler = MinMaxScaler()
	numeric_features = ['quantity', 'unit_price', 'total_contract_amount']
	filtered_df[numeric_features] = scaler.fit_transform(filtered_df[numeric_features])
	
	# Выбор признаков для кластеризации
	features = filtered_df[['quantity', 'unit_price', 'total_contract_amount']]
	
	# Построение k-NN графика
	plot_knn_graph(features, k=5)
	
	# Выполнение DBSCAN
	eps = 0.15  # Оптимальное значение можно выбрать после анализа графика
	min_samples = 5
	clustered_data = perform_dbscan(features, eps=eps, min_samples=min_samples)
	
	# Анализ результатов
	clusters_summary = clustered_data.groupby('cluster').mean()
	print(f"Количество точек шума: {len(clustered_data[clustered_data['cluster'] == -1])}")
	print(f"Количество кластеров: {len(clustered_data['cluster'].unique()) - 1}")  # -1 для исключения шума
	
	return clustered_data, clusters_summary

# Запуск обработки
# clustered_data, clusters_summary = main(contract_df)