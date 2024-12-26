import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox
from sklearn.ensemble import IsolationForest
from utils.vizualization_tools import plot_bar_chart, visualize_isolation_forest
from utils.functions import CurrencyConverter
from sklearn.neighbors import LocalOutlierFactor
import os


# 2. Анализ эффективности исполнителей
def analyze_efficiency(filtered_df):
	print('Запускается метод analyze_efficiency')
	print(filtered_df.columns)
	
	columns_info = [
		('unit_price', 'currency', 'unit_price_in_eur'),
		('total_price', 'currency', 'total_price_in_eur')
	]
	converter = CurrencyConverter()
	filtered_df = converter.convert_multiple_columns(filtered_df, columns_info=columns_info)
	
	results = []
	
	# Вызываем Isolation Forest для аномалий
	model = IsolationForest(contamination=0.05, random_state=42)
	filtered_df['is_anomaly'] = model.fit_predict(
		filtered_df[['unit_price_in_eur', 'total_price_in_eur', 'supplier_qty']])
	
	# Аномалии: -1, нормальные: 1
	filtered_df['is_anomaly'] = filtered_df['is_anomaly'].map({1: False, -1: True})
	
	# убедимся, что is_anomaly возвращает непустые группы
	print("Grouped data by 'is_anomaly':")
	print(filtered_df.groupby('is_anomaly').size())
	
	# Распределение цены за единицу
	try:
		plt.figure(figsize=(10, 6))
		grouped = filtered_df.groupby('is_anomaly')
		for is_anomaly, group in grouped:
			if not group.empty:
				group['total_price_in_eur'].plot(kind='hist', bins=30, alpha=0.5, label=f"Anomaly={is_anomaly}")
		
		plt.title('Distribution of Total Prices in EUR')
		plt.xlabel('Total Price (EUR)')
		plt.ylabel('Frequency')
		plt.legend()
		plt.grid(True)  # добавляет сетку для удобства
		plt.tight_layout()  # Убирает лишние отступы
		plt.show()
	except Exception as e:
		print(f"Error during plot:{e}")
	
	# Статистика
	stats = filtered_df.groupby('is_anomaly').agg({
		'unit_price_in_eur': ['median', 'mean', 'std'],
		'total_price_in_eur': ['median', 'mean', 'std'],
		'supplier_qty': ['median', 'mean', 'std']
	})
	
	print("Statistics by anomaly status:")
	print(stats)
	
	return filtered_df, stats


# детальный анализ аномальных лотов
def detailed_anomaly_analysis(analyzed_df):
	"""
	Метод для анализа аномальных лотов из analyzed_df.
	"""
	print("Запущен detailed_anomaly_analysis")
	
	# 1. Проверяем наличие аномалий
	anomalous_lots = analyzed_df[analyzed_df['is_anomaly'] == True]
	
	if anomalous_lots.empty:
		print("Нет аномальных данных для анализа.")
		return None
	
	# 2. Подготовка данных для анализа
	anomalous_summary = anomalous_lots[['lot_number', 'unit_price_in_eur', 'total_price_in_eur', 'supplier_qty',
	                                    'actor_name', 'winner_name']].sort_values(by='total_price_in_eur',
	                                                                              ascending=False)
	
	# Группировка по исполнителям и поставщикам
	actors_analysis = anomalous_lots.groupby('actor_name').size().sort_values(ascending=False).reset_index(
		name="Anomalous Lots Count")
	winners_analysis = anomalous_lots.groupby('winner_name').size().sort_values(ascending=False).reset_index(
		name="Anomalous Lots Count")
	
	# 2.1 визуализация actors_analysis и winners_analysis
	
	top_winners = winners_analysis.nlargest(20, "Anomalous Lots Count")  # Топ-10 победителей
	plt.figure(figsize=(16, len(top_winners) * 0.5))
	sns.barplot(
		data=top_winners,
		x="Anomalous Lots Count",
		y="winner_name",
		orient="h",
		palette="viridis",
		dodge=False
	)
	plt.title("Top 20 Anomalous Lots Count by Winner")
	plt.xlabel("Anomalous Lots Count")
	plt.ylabel("Winner Name")
	plt.tight_layout()
	plt.show()
	
	top_actors = actors_analysis.nlargest(20, "Anomalous Lots Count")  # Топ-10 исполнителей
	plt.figure(figsize=(16, len(top_actors) * 0.5))
	sns.barplot(
		data=top_actors,
		x="Anomalous Lots Count",
		y="actor_name",
		orient="h",
		palette="viridis",
		dodge=False
	)
	plt.title("Top 20 Anomalous Lots Count by Actor")
	plt.xlabel("Anomalous Lots Count")
	plt.ylabel("Actor Name")
	plt.tight_layout()
	plt.show()
	
	# 3. Сохранение результатов
	output_folder = 'D:\Analysis-Results\efficient_analyses'
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	
	output_file = f"{output_folder}\Anomalous_Lots_Analysis.xlsx"
	with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
		anomalous_summary.to_excel(writer, sheet_name='Anomalous Lots', index=False)
		actors_analysis.to_excel(writer, sheet_name='Actors Analysis', index=False)
		winners_analysis.to_excel(writer, sheet_name='Winners Analysis', index=False)
	
	print(f"Результаты анализа аномалий сохранены в файл: {output_file}")
	return anomalous_summary, actors_analysis, winners_analysis


# вызов функций из главного метода
def main_method(filtered_df, data_df, parent_widget=None):
	print("Мы вошли в метод main_method")
	output_folder = 'D:\Analysis-Results\efficient_analyses'
	print(filtered_df.columns)
	
	# добираем товары выбранной категории в датафрейм
	selected_lots = filtered_df['lot_number'].unique()
	filtered_df = data_df[data_df['lot_number'].isin(selected_lots)]
	
	analyzed_df, stats = analyze_efficiency(filtered_df)
	
	# Сохранение анализа эффективности исполнителей
	file_path = f"{output_folder}\Efficiency_Metrics.xlsx"
	print(f"Сохраняем файл в: {file_path}")
	
	if not analyzed_df.empty:
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)
		
		file_saved = False
		while not file_saved:
			try:
				# Сохраняем основной анализ
				analyzed_df.to_excel(f"{output_folder}\Efficiency_Metrics.xlsx", index=False)
				print("Анализ эффективности исполнителей сохранен в файл 'Efficiency_Metrics.xlsx'")
				
				# Сохраняем статистику по аномалиям
				with pd.ExcelWriter(f"{output_folder}\Efficiency_Metrics.xlsx", engine='openpyxl', mode='a') as writer:
					stats.to_excel(writer, sheet_name='Anomaly_Stats')
				print("Статистика по аномалиям добавлена в файл 'Efficiency_Metrics.xlsx' на лист 'Anomaly_Stats'.")
				
				file_saved = True
			except PermissionError:
				print("Файл используется другой программой.")
				# Показать предупреждение пользователю
				msg_box = QMessageBox(parent_widget)
				msg_box.setIcon(QMessageBox.Warning)
				msg_box.setWindowTitle("Файл используется")
				msg_box.setText(
					"Файл 'Efficiency_Metrics.xlsx' уже используется другой программой.\nПожалуйста, закройте файл и нажмите OK для продолжения.")
				msg_box.setStandardButtons(QMessageBox.Ok)
				msg_box.exec_()
	else:
		print("Нет данных для анализа эффективности исполнителей для сохранения.")
	
	# занимаемся анализом аномальных лотов
	detailed_anomaly_analysis(analyzed_df)
	
	# Визуализация Isolation Foresf
	visualize_isolation_forest(analyzed_df)
	
	return analyzed_df


def display_dataframe_to_user(name, dataframe):
	"""
	Отображает DataFrame в PyQt таблице.
	:param name: Название таблицы.
	:param dataframe: DataFrame для отображения.
	"""
	from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QLabel, QDialog
	import pandas as pd
	
	# Создаём диалоговое окно
	dialog = QDialog()
	dialog.setWindowTitle(name)
	
	# Создаём таблицу
	table = QTableWidget()
	table.setRowCount(len(dataframe))
	table.setColumnCount(len(dataframe.columns))
	table.setHorizontalHeaderLabels(dataframe.columns)
	
	# Заполняем таблицу данными
	for i, row in enumerate(dataframe.itertuples(index=False)):
		for j, value in enumerate(row):
			table.setItem(i, j, QTableWidgetItem(str(value)))
	
	# Добавляем виджеты в диалог
	layout = QVBoxLayout()
	layout.addWidget(QLabel(f"<h3>{name}</h3>"))
	layout.addWidget(table)
	dialog.setLayout(layout)
	
	# Показываем окно
	dialog.exec_()


def analyze_suppliers_by_unit_price(parent_widget, mydata_df, data_df):
	print("Загружен метод analyze_suppliers_by_unit_price")
	"""
	  Анализ поставщиков по средней цене за единицу товаров.
	  :param mydata_df: DataFrame с товарами для анализа
	  :param data_df: Полный DataFrame для фильтрации лотов
	"""
	output_folder = 'D:\Analysis-Results\suppliers_by_unit_price'
	
	# Создаём папку, если она не существует
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	
	# Приведение валют к единой
	print("Приведение валют к единой валюте (EUR)...")
	converter = CurrencyConverter
	mydata_df = converter.convert_column(
		mydata_df,
		amount_column='unit_price',
		currency_column='currency'
	)
	
	# Удаляем лишние пробелы и приводим к нижнему регистру
	mydata_df['good_name'] = mydata_df['good_name'].str.strip().str.lower()
	
	# Частота появления товаров
	goods_frequency = mydata_df['good_name'].value_counts()
	repeated_goods = goods_frequency[goods_frequency > 1]
	
	if repeated_goods.empty:
		QMessageBox.information(parent_widget, "Результат", "Нет товаров, которые встречаются в нескольких лотах.")
		return None
	
	# Создаём итоговую таблицу
	result_table = []
	
	# Обрабатываем повторяющиеся товары
	for good_name in repeated_goods.index:
		# Фильтруем данные для текущего товара
		filtered_goods = mydata_df[mydata_df['good_name'] == good_name]
		
		# Группируем по поставщикам
		supplier_data = (
			filtered_goods.groupby('winner_name')
			.agg(
				avg_unit_price=('converted_amount', 'mean'),
				lot_numbers=('lot_number', lambda x: ', '.join(map(str, x.unique()))),
				lots_count=('lot_number', 'nunique')
			)
			.reset_index()
		)
		
		# Добавляем информацию о товаре в таблицу
		supplier_data['good_name'] = good_name
		result_table.append(supplier_data)
	
	# Объединяем все данные в одну таблицу
	final_table = pd.concat(result_table, ignore_index=True)
	
	# Сортируем по товару и средней цене
	final_table = final_table.sort_values(by=['good_name', 'avg_unit_price'], ascending=[True, True])
	
	# Сохраняем таблицу в Excel
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	excel_file = os.path.join(output_folder, f"supplier_analysis_{timestamp}.xlsx")
	final_table.to_excel(excel_file, index=False)
	# print(f"Результаты сохранены в файл: {excel_file}")
	
	# Визуализация
	for good_name in repeated_goods.index:
		filtered_data = final_table[final_table['good_name'] == good_name]
		
		# Создаём путь для графика
		png_file = os.path.join(output_folder, f"{good_name}_price_analysis.png".replace("/", "_"))
		
		# Используем функцию визуализации из модуля
		plot_bar_chart(
			x=filtered_data['winner_name'],
			y=filtered_data['avg_unit_price'],
			title=f'Средняя цена за единицу для товара: {good_name}',
			x_label='Поставщик',
			y_label='Средняя цена за единицу (EUR)',
			output_file=png_file
		)
	print('Мы вышли из plot_bar_chart')
	# Отображаем сообщение об успешном завершении
	QMessageBox.information(parent_widget, "Результат", f"Все данные выведены в папку '{output_folder}'")
	
	return
