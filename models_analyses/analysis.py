import pandas as pd
import os
import gc
import networkx as nx
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMessageBox
from utils.vizualization_tools import save_top_suppliers_bar_chart
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


def analyze_monthly_expenses(df, start_date, end_date):
	# Создаем папку для результатов, если её еще нет
	output_dir = "D:/Analysis-Results"
	os.makedirs(output_dir, exist_ok=True)
	
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
	missing_currencies = set(df['currency'].unique()) - set(grouped.columns)
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
	filename = f"monthly_expenses_{filter_description}_normalized.png"
	filepath = os.path.join(output_dir, filename)
	plt.savefig(filepath, bbox_inches='tight')
	plt.show()
	
	print(f"Нормализованный объединенный график успешно сохранен как '{filepath}'.")


def analyze_top_suppliers(parent_widget, df, start_date, end_date):
	# Создаем папку для результатов, если её еще нет
	output_dir = "D:/Analysis-Results/Top_10"
	os.makedirs(output_dir, exist_ok=True)
	# Группируем данные по валютам
	grouped_by_currency = group_by_currency(df)
	
	# Вычисляем количество лет и месяцев между датами
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
	for currency, group in grouped_by_currency:
		# Вывод информации о текущей группе
		print(f"Валюта: {currency}, количество записей: {len(group)}")
		
		# Проверка наличия данных
		if group.empty:
			print(f"Нет данных для валюты {currency}, пропускаем...")
			continue
		
		# Группировка по поставщикам и подсчет затрат
		top_suppliers = group.groupby('winner_name')['total_price'].sum().nlargest(10)
		
		print(f"Топ-10 поставщиков для валюты {currency}:")
		print(top_suppliers)  # Проверочный вывод
		
		# Проверка наличия данных после группировки
		if top_suppliers.empty:
			print(f"Нет данных для построения графика по валюте {currency}.")
			continue
		
		# вызываем функцию визуализации
		save_top_suppliers_bar_chart(top_suppliers, currency, interval_text, output_dir)
	
	QMessageBox.information(parent_widget, "Результат",
	                        f"Анализ Топ-10 поставщиков завершен. Графики сохранены в папке "
	                        f"{output_dir} ")


""" -------------------------------------------------------------- """

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
			
			print(f"График сохранен: {file_path}")
	
	return top_suppliers


def network_analysis(parent_widget, df):
	"""
	:param df: отфильтрованный датафрейм по одному проекту
	:return: None
	Использован алгоритм Fruchterman-Reingold Algorithm и
	kamada_kawai - алгоритм упругих взаимодействий
	"""
	print('Запускается метод сетевого анализа и построения графов')
	
	# Преобразуем значения project_name в строки
	df['project_name'] = df['project_name'].astype(str)
	
	# Извлечение уникальных значений для валют
	unique_currencies = df['currency'].unique().tolist()
	selected_project = df['project_name'].unique()[0]
	
	output_folder = 'D:/Analysis-Results/network_graphs'
	os.makedirs(output_folder, exist_ok=True)
	
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
		
		# Оптимизация размещения узлов (алгоритм spring_layout)
		pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
		
		# получение цветов узлов из атрибутов
		node_colors = [data['color'] for _, data in G.nodes(data=True)]
		
		# Визуализация сети
		plt.figure(figsize=(15, 10))
		nx.draw(G, pos, with_labels=True, node_size=700, node_color=node_colors, font_size=6, font_color='black',
		        edge_color='gray')
		plt.title(f'Network for {selected_project} in {currency}')
		
		# Сохранение графика в файл
		file_path = os.path.join(output_folder, f'network_{selected_project}_{currency}.png')
		plt.savefig(file_path)
		plt.close('all')
		gc.collect()
	
	# Создание объединенной сети для всех валют
	G_combined = nx.Graph()
	G_combined.add_nodes_from(df['discipline'].unique(), type='discipline', color='green')
	G_combined.add_nodes_from(df['winner_name'].unique(), type='supplier', color='lightblue')
	G_combined.add_node(selected_project, type='project', color='red')  # Узел для проекта
	
	# Добавление связей на основе всех данных проекта
	for _, row in df.iterrows():
		discipline = row['discipline']
		supplier = row['winner_name']
		
		# Проверка на наличие дисциплины и поставщика
		if pd.notna(discipline) and pd.notna(supplier):
			# Добавление связей (ребер) между узлами
			G_combined.add_edge(selected_project, discipline)  # Связь проект - дисциплина
			G_combined.add_edge(discipline, supplier)  # Связь дисциплина - поставщик
	
	# Список различных алгоритмов размещения
	layouts = {
		'spring': nx.spring_layout,
		'kamada_kawai': nx.kamada_kawai_layout,
	}
	
	# Перебор всех вариантов размещения
	for layout_name, layout_func in layouts.items():
		print(f"Построение графика с размещением: {layout_name}")
		# вычисление координат узлов
		try:
			pos_combined = layout_func(G_combined, seed=42) if layout_name == 'spring' else layout_func(G_combined)
		except Exception as e:
			print(f"Ошибка при вычислении layout {layout_name}: {e}")
			continue
		
		# Получение цветов узлов из атрибутов
		node_colors_combined = [data['color'] for _, data in G_combined.nodes(data=True)]
		
		# Визуализация сети с выбранным размещением
		plt.figure(figsize=(15, 10))
		nx.draw(G_combined, pos_combined, with_labels=False, node_size=700, node_color=node_colors_combined,
		        edge_color='gray')
		nx.draw_networkx_labels(G_combined, pos_combined, font_size=8, font_color='black')
		plt.title(f'Combined Network for {selected_project} (All Currencies) - {layout_name.capitalize()} Layout')
		
		# Сохранение графика в файл в указанной папке
		file_path = os.path.join(output_folder, f'combined_network_{selected_project}_all_currencies_{layout_name}.png')
		try:
			plt.savefig(file_path)
			print(f"Объединенный график с размещением {layout_name} сохранен: {file_path}")
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
	print(df_converted.head(10))
	
	results = []
	
	# Перебор всех строк в common_suppliers_df
	for _, row in common_suppliers_df.iterrows():
		discipline1 = row['discipline1']
		discipline2 = row['discipline2']
		common_suppliers = row['common_suppliers']
		
		for supplier in common_suppliers:
			# Фильтруем данные для поставщика в обеих дисциплинах
			discipline1_data = df_converted[(df_converted['discipline'] == discipline1) & (df_converted['winner_name'] == supplier)]
			discipline2_data = df_converted[(df_converted['discipline'] == discipline2) & (df_converted['winner_name'] == supplier)]
			
			for good_name in set(discipline1_data['good_name']).intersection(set(discipline2_data['good_name'])):
				discipline1_goods = discipline1_data[discipline1_data['good_name'] == good_name]
				discipline2_goods = discipline2_data[discipline2_data['good_name'] == good_name]
				
				price1 = discipline1_goods['amount_eur'].mean()
				price2 = discipline2_goods['amount_eur'].mean()
				
				# Извлекаем номера лотов
				lot_numbers_discipline1 = discipline1_goods['lot_number'].unique()
				lot_numbers_discipline2 = discipline2_goods['lot_number'].unique()
				
				persent_of_difference = (price1 - price2)*100/(price1+price2)
				# if persent_of_difference != 0 and abs(persent_of_difference) > 10:
				if persent_of_difference > 10:
					results.append({
						'supplier': supplier,
						'good_name': good_name,
						'discipline1': discipline1,
						'discipline2': discipline2,
						'price_discipline1': price1,
						'price_discipline2': price2,
						'price_difference': persent_of_difference,
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
		print(f"Файл успешно сохранён: {file_path}")
	else:
		print("Файл занят, программа не может продолжить работу")
	return results_df
	
	# Сохраняем DataFrame в Excel
	results_df.to_excel(file_path, index=False)
	


def matches_results_stat(comparison_results):
	# общее количество совпадений
	total_matches = len(comparison_results)
	unique_suppliers = comparison_results['supplier'].nunique()
	
	# Средний процент расхождения цен
	average_difference = comparison_results['price_difference'].mean()
	
	# Топ-10 поставщиков по количеству совпадений
	top_suppliers = comparison_results['supplier'].value_counts().head(10)
	
	
	
	# Вывод статистики
	print(f"Общее количество совпадений: {total_matches}")
	print(f"Уникальные поставщики: {unique_suppliers}")
	print(f"Средний процент расхождения цен: {average_difference:.2f}%")
	print("Топ-10 поставщиков по количеству совпадений:")
	print(top_suppliers)