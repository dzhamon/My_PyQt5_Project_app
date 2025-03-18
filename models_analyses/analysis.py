import pandas as pd
from datetime import datetime
import os
import gc
import networkx as nx
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QMetaObject, Qt
from utils.vizualization_tools import save_top_suppliers_bar_chart
from utils.config import BASE_DIR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def process_data(df):
	# предобработка данных
	df['total_price'] = pd.to_numeric(df['total_price'], errors='coerce')
	df = df.dropna(subset=['close_date', 'total_price'])
	# Преобразуем 'close_date' в формат datetime
	df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')
	
	# Удаляем строки с NaT в 'close_date'
	df = df.dropna(subset=['close_date'])
	return df


def group_by_currency(df):
	# Группировка данных по валютам
	grouped = df.groupby('currency')
	return grouped


def analyze_monthly_cost(parent_widget, df, start_date, end_date):
	"""
	Для полноценного анализа месяных затрат на закупку материалов
	необходимы данные по планируемому бюджету закупок на этот проект.
	Причем, необходимо иметь данные бюджета по различным категориям материалов.
	:param df:
	:param start_date:
	:param end_date:
	:return:
	"""
	# Создаем папку для результатов, если её еще нет
	OUT_DIR = os.path.join(BASE_DIR, "monthly_cost")
	os.makedirs(OUT_DIR, exist_ok=True)
	
	# Конвертация дат
	df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')
	filtered_df = df[(df['close_date'] >= start_date) & (df['close_date'] <= end_date)]
	
	# Проверка данных после фильтрации
	if filtered_df.empty:
		print("Нет данных для заданного диапазона дат.")
		return
	
	# Определение фильтра для заголовка и имени файла
	if 'discipline' in df.columns and df['discipline'].nunique() == 1:
		filter_by = 'discipline'
		filter_value = df['discipline'].iloc[0]
	elif 'project_name' in df.columns and df['project_name'].nunique() == 1:
		filter_by = 'project_name'
		filter_value = df['project_name'].iloc[0]
		print('Отфильтрован по', filter_value)
	else:
		filter_by = 'all_data'
		filter_value = None
	
	filter_description = f"{filter_by}_{filter_value}" if filter_value else "all_data"
	
	# Создаем столбец year_month для группировки по месяцам
	filtered_df['year_month'] = filtered_df['close_date'].dt.to_period('M')
	
	# Группировка по валюте и месяцу
	grouped = filtered_df.groupby(['year_month', 'currency'])['total_price'].sum().unstack()
	
	# Проверка на наличие данных для каждой валюты
	missing_currencies = set(df['currency'].unique()) - set(filtered_df['currency'].unique())
	if missing_currencies:
		print(f"Отсутствуют данные для валют: {missing_currencies}")
	
	# Нормализация данных по каждой валюте
	normalized_grouped = grouped.apply(lambda x: (x / x.max()) * 100 if x.max() != 0 else x)
	
	# Построение объединенного графика по всем валютам с нормализованными значениями
	plt.figure(figsize=(14, 8))
	normalized_grouped.plot(kind='line', marker='o', ax=plt.gca())
	
	plt.title(
		f'Нормализованная динамика месячных затрат по валютам за период {start_date} - {end_date}\n(Фильтр: {filter_description})')
	plt.xlabel('Месяц')
	plt.ylabel('Затраты (% от максимального значения)')
	plt.grid(True)
	plt.legend(title='Валюта', loc='upper left', bbox_to_anchor=(1, 1))
	
	# Добавление подписей значений для каждой валюты
	for currency in normalized_grouped.columns:
		for x, y in zip(normalized_grouped.index, normalized_grouped[currency]):
			if pd.notna(y):
				plt.text(x.ordinal, y, f'{y:.0f}%', ha='center', va='bottom')
	
	# Сохранение графика в файл с идентификатором фильтра
	filename = f"monthly_cost_{filter_description}_norm.png"
	filepath = os.path.join(OUT_DIR, filename)
	plt.savefig(filepath, bbox_inches='tight')
	plt.show()
	
	QMessageBox.information(parent_widget, "Результат",
	                        f"Анализ месячных затрат завершен. Графики сохранены в папке "
	                        f"{OUT_DIR} ")


def analyze_top_suppliers(parent_widget, df, start_date, end_date, project_name):
	"""
	   Анализирует топ-10 поставщиков и сохраняет результаты в папку, названную по имени проекта.

	   :param parent_widget: Родительский виджет для отображения сообщений.
	   :param df: DataFrame с данными.
	   :param start_date: Начальная дата периода анализа.
	   :param end_date: Конечная дата периода анализа.
	   :param project_name: Наименование проекта.
	   """
	# Создаем папку для результатов, если её еще нет
	OUT_DIR  = os.path.join(BASE_DIR, "top_10_suppls", project_name)
	os.makedirs(OUT_DIR, exist_ok=True)
	
	# Группируем данные по валютам
	grouped_by_currency = group_by_currency(df)
	
	# Вычисляем количество лет и месяцев между датами
	start_date = datetime.strptime(start_date, "%Y-%m-%d")
	end_date = datetime.strptime(end_date, "%Y-%m-%d")
	delta = end_date - start_date
	num_years = delta.days // 365
	num_months = (delta.days % 365) // 30
	
	# Формируем текстовый интервал для заголовка
	interval_text = ''
	if num_years > 0:
		interval_text += f'{num_years} года' if num_years == 1 else f'{num_years} лет'
	if num_months > 0:
		if interval_text:
			interval_text += ' и '
		interval_text += f'{num_months} месяца' if num_months == 1 else f'{num_months} месяцев'
	
	# Проходим по каждой группе валют
	# и создаем excel-файл с помощью ExcelWriter
	file_exls_name = f"top_10_report_{project_name}.xlsx"
	file_exls_path = os.path.join(OUT_DIR, file_exls_name)
	with pd.ExcelWriter(file_exls_path, engine='xlsxwriter') as writer:
		for currency, group in grouped_by_currency:
			# Вывод информации о текущей группе
			print(f"Валюта: {currency}, количество записей: {len(group)}")
			
			# Проверка наличия данных
			if group.empty:
				print(f"Нет данных для валюты {currency}, пропускаем...")
				continue
			
			# Группировка по поставщикам и подсчет затрат
			top_suppliers = group.groupby('winner_name')['total_price'].sum().nlargest(10)
			
			# Проверка наличия данных после группировки
			if top_suppliers.empty:
				print(f"Нет данных для построения графика по валюте {currency}.")
				continue
			# Преобразуем Series в DataFrame для записи в Excel
			top_suppliers_df = top_suppliers.reset_index()
			top_suppliers_df.columns = ['Supplier', 'Total Costs']
			
			# Записываем данные в Excel на отдельный лист
			sheet_name = f"Top Suppliers ({currency})"
			top_suppliers_df.to_excel(writer, sheet_name=sheet_name, index=False)
			
			# Добавляем форматирование
			workbook = writer.book
			worksheet = writer.sheets[sheet_name]
			
			# Форматируем заголовки
			header_format = workbook.add_format({
				'bold': True,
				'text_wrap': True,
				'valign': 'top',
				'fg_color': '#4F81BD',
				'font_color': '#FFFFFF',
				'border': 1
			})
			
			# Форматируем ячейки с данными
			data_format = workbook.add_format({
				'num_format': '#,##0',
				'border': 1
			})
			
			# Применяем форматирование к заголовкам
			for col_num, value in enumerate(top_suppliers_df.columns.values):
				worksheet.write(0, col_num, value, header_format)
			
			# Применяем форматирование к данным
			for row_num, data in enumerate(top_suppliers_df.values, start=1):
				for col_num, value in enumerate(data):
					if col_num == 1:  # Форматируем столбец с суммами
						worksheet.write(row_num, col_num, value, data_format)
					else:
						worksheet.write(row_num, col_num, value)
			
			# Автонастройка ширины столбцов
			worksheet.set_column('A:B', 20)
			
			# вызываем функцию визуализации
			save_top_suppliers_bar_chart(top_suppliers, currency, interval_text, OUT_DIR)
		
	QMessageBox.information(parent_widget, "Результат",
	                        f"Анализ Топ-10 поставщиков завершен. Графики и файл {file_exls_name}"
                            f"сохранены в папке: {OUT_DIR}")


# -----------------------------------------------------

""" Анализ Частота появления Поставщика"""


# DataFrame называется df и содержит столбцы 'discipline', 'actor_name', 'winner_name'

def analyze_supplier_frequency(df, output_dir="D:/Analysis-Results/Supplier-Frequency", threshold=1):
	import os
	import matplotlib.pyplot as plt
	os.makedirs(output_dir, exist_ok=True)
	
	# Группировка данных
	grouped_df = df.groupby(['discipline', 'actor_name', 'winner_name']).size().reset_index(name='win_count')
	
	# Сортировка данных
	top_suppliers = grouped_df.sort_values(by=['discipline', 'actor_name', 'win_count'], ascending=[True, True, False])
	
	# Вывод информации
	print("Частота выигрышей поставщиков по дисциплинам и исполнителям:")
	print(top_suppliers)
	
	# Визуализация
	for discipline, group_discipline in top_suppliers.groupby('discipline'):
		# Создаем папку для текущей дисциплины
		discipline_dir = os.path.join(output_dir, discipline.replace(" ", "_"))
		os.makedirs(discipline_dir, exist_ok=True)
		# Сохраняем данные дисциплины в Excel
		excel_path = os.path.join(discipline_dir, f"{discipline}_supplier_frequency.xlsx")
		with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
			group_discipline.to_excel(writer, index=False, sheet_name='Supplier Frequency')
		print(f"Сохранен файл Excel: {excel_path}")
		
		# Построение графиков для каждой пары (discipline, actor_name)
		for actor_name, group_actor in group_discipline.groupby('actor_name'):
			# Фильтрация по threshold
			group_actor = group_actor[group_actor['win_count'] >= threshold]
			if group_actor.empty:
				continue
			
			# Построение графика
			fig, ax = plt.subplots(figsize=(12, 8))
			ax.bar(group_actor['winner_name'], group_actor['win_count'], color='skyblue')
			ax.set_title(f'{discipline} ({actor_name})')
			ax.set_xlabel('Supplier')
			ax.set_ylabel('Win Count')
			ax.tick_params(axis='x', rotation=45)
			plt.tight_layout()
			
			# Сохранение графика
			file_path = os.path.join(discipline_dir, f"{actor_name}_supplier_frequency.png")
			plt.savefig(file_path)
			plt.close()
	return top_suppliers


def network_analysis(parent_widget, df):
	"""
	:param df: отфильтрованный датафрейм по одному проекту
	:return: None
	Использованы алгоритмы Fruchterman-Reingold Algorithm и kamada_kawai
	"""
	print('Запускается метод сетевого анализа и построения графов')
	
	# Преобразуем значения project_name в строки
	df['project_name'] = df['project_name'].astype(str)
	
	# Извлечение уникальных значений для валют
	unique_currencies = df['currency'].unique().tolist()
	selected_project = df['project_name'].unique()[0]
	
	output_folder = 'D:/Analysis-Results/network_graphs'
	os.makedirs(output_folder, exist_ok=True)
	
	# Список алгоритмов размещения
	layouts = {
		'spring': nx.spring_layout,
		'kamada_kawai': nx.kamada_kawai_layout,
	}
	
	# Перебираем все уникальные валюты для данного проекта
	for currency in unique_currencies:
		# Фильтрация данных по валюте
		currency_data = df[df['currency'] == currency]
		
		# Извлечение уникальных дисциплин и поставщиков для текущей валюты
		unique_disciplines = currency_data['discipline'].unique().tolist()
		unique_suppliers = currency_data['winner_name'].unique().tolist()
		
		# Создание пустого графа для текущей валюты
		G = nx.Graph()
		
		# Добавление узла для проекта (красный цвет)
		G.add_node(selected_project, type='project', color='red')
		
		# Добавление узлов для дисциплин и поставщиков только для текущей валюты
		G.add_nodes_from(unique_disciplines, type='discipline', color='green')
		G.add_nodes_from(unique_suppliers, type='supplier', color='lightblue')
		
		# Добавление связей на основе данных проекта и текущей валюты
		for _, row in currency_data.iterrows():
			discipline = row['discipline']
			supplier = row['winner_name']
			
			if pd.notna(discipline) and pd.notna(supplier):
				# Добавляем связь проект - дисциплина
				G.add_edge(selected_project, discipline)
				# Добавляем связь дисциплина - поставщик
				G.add_edge(discipline, supplier)
		
		# Перебираем все алгоритмы размещения
		for layout_name, layout_func in layouts.items():
			print(f"Построение графика для {currency} с размещением: {layout_name}")
			
			# Оптимизация размещения узлов
			try:
				pos = layout_func(G, seed=42) if layout_name == 'spring' else layout_func(G)
			except Exception as e:
				print(f"Ошибка при вычислении layout {layout_name}: {e}")
				continue
			
			# Получение цветов узлов из атрибутов
			node_colors = [data['color'] for _, data in G.nodes(data=True)]
			
			# Визуализация сети
			plt.figure(figsize=(15, 10))
			nx.draw(G, pos, with_labels=True, node_size=700, node_color=node_colors, font_size=6, font_color='black',
			        edge_color='gray')
			# Заголовок
			title = f'Network for {selected_project} in {currency} - {layout_name.capitalize()} Layout'
			plt.title(title, fontsize=14)
			
			# Расширение области для добавления пояснения
			plt.subplots_adjust(bottom=0.2)
			
			# Добавление пояснения
			description = f"Project: {selected_project}, Currency: {currency}, Layout: {layout_name.capitalize()}"
			plt.figtext(0.5, 0.02, description, wrap=True, horizontalalignment='center', fontsize=10)
			
			# Сохранение графика в файл
			file_path = os.path.join(output_folder, f'network_{selected_project}_{currency}_{layout_name}.png')
			try:
				plt.savefig(file_path)
				print(f"График с размещением {layout_name} сохранен: {file_path}")
			except Exception as error:
				print(f"Ошибка при сохранении графика {layout_name}: {error}")
			finally:
				plt.close('all')
				gc.collect()
	
	QMessageBox.information(parent_widget, "Сообщение",
	                        f"Метод сетевого анализа завершен. Файлы сохранены в папке {output_folder}")
	return


def find_common_suppliers_between_disciplines(df):
	"""
	Проверяет, имеют ли поставщики одной дисциплины общих поставщиков с другой дисциплиной
	и возвращает номера лотов для этих поставщиков.
	Параметры:
		df (DataFrame): Данные с колонками ['discipline', 'winner_name', 'lot_number'].
	Возвращает:
		DataFrame: Таблица с парами дисциплин, списком общих поставщиков и номерами лотов.
	"""
	# Группировка данных по дисциплинам
	discipline_suppliers = df.groupby('discipline')['winner_name'].apply(set)
	
	# Список всех дисциплин
	disciplines = discipline_suppliers.index.tolist()
	
	# Список для результатов
	results = []
	
	# Перебор всех пар дисциплин
	for i, discipline1 in enumerate(disciplines):
		for discipline2 in disciplines[i + 1:]:
			# Найдем общих поставщиков между дисциплинами
			common_suppliers = discipline_suppliers[discipline1] & discipline_suppliers[discipline2]
			
			# Если есть общие поставщики, формируем результирующий список
			if common_suppliers:
				results.append({
					'discipline1': discipline1,
					'discipline2': discipline2,
					'common_suppliers': list(common_suppliers)
				})
	
	# Преобразование результатов в DataFrame
	return pd.DataFrame(results)


def compare_materials_and_prices(df, common_suppliers_df):
	from utils.functions import CurrencyConverter, check_file_access
	
	converter = CurrencyConverter()
	df_converted = converter.convert_column(df, amount_column='unit_price', currency_column='currency',
	                                        result_column='amount_eur')
	
	results = []
	
	# Перебор всех строк в common_suppliers_df
	for _, row in common_suppliers_df.iterrows():
		discipline1 = row['discipline1']
		discipline2 = row['discipline2']
		common_suppliers = row['common_suppliers']
		
		for supplier in common_suppliers:
			# Фильтруем данные для поставщика в обеих дисциплинах
			discipline1_data = df_converted[
				(df_converted['discipline'] == discipline1) & (df_converted['winner_name'] == supplier)]
			discipline2_data = df_converted[
				(df_converted['discipline'] == discipline2) & (df_converted['winner_name'] == supplier)]
			
			for good_name in set(discipline1_data['good_name']).intersection(set(discipline2_data['good_name'])):
				discipline1_goods = discipline1_data[discipline1_data['good_name'] == good_name]
				discipline2_goods = discipline2_data[discipline2_data['good_name'] == good_name]
				
				price1 = discipline1_goods['amount_eur'].mean()
				price2 = discipline2_goods['amount_eur'].mean()
				
				# Извлекаем номера лотов
				lot_numbers_discipline1 = discipline1_goods['lot_number'].unique()
				lot_numbers_discipline2 = discipline2_goods['lot_number'].unique()
				
				persent_of_difference = (price1 - price2) * 100 / (price1 + price2)
				
				if persent_of_difference > 10 or persent_of_difference < -10:
					results.append({
						'supplier': supplier,
						'good_name': good_name,
						'discipline1': discipline1,
						'discipline2': discipline2,
						'price_discipline1': price1,
						'price_discipline2': price2,
						'persent_of_diff': persent_of_difference,
						'lot_numbers_discipline1': lot_numbers_discipline1.tolist(),
						'lot_number_discipline2': lot_numbers_discipline2.tolist()
					})
	# Преобразуем results в DataFrame
	results_df = pd.DataFrame(results)
	
	# Указание папки и имени файла
	output_folder = r"D:\Analysis-Results\suppliers_between_disciplines"
	os.makedirs(output_folder, exist_ok=True)
	file_path = os.path.join(output_folder, "suppliers_analysis.xlsx")
	
	if check_file_access(file_path):
		# Сохраняем DataFrame в Excel
		results_df.to_excel(file_path, index=False)
		print(f"Файл успешно сохранён: {file_path}")
	else:
		print("Файл занят, программа не может продолжить работу")
	
	return results_df

def matches_results_stat(comparison_results):
	# общее количество совпадений
	total_matches = len(comparison_results)
	unique_suppliers = comparison_results['supplier'].nunique()
	
	# Средний процент расхождения цен
	average_difference = comparison_results['persent_of_diff'].mean()
	
	# Топ-10 поставщиков по количеству совпадений
	top_suppliers = comparison_results['supplier'].value_counts().head(10)
	
	# Вывод статистики
	print(f"Общее количество совпадений: {total_matches}")
	print(f"Уникальные поставщики: {unique_suppliers}")
	print(f"Средний процент расхождения цен: {average_difference:.2f}%")
	print("Топ-10 поставщиков по количеству совпадений:")
	print(top_suppliers)
