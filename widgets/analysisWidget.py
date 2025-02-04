# analysis_widget.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton
import pandas as pd
from utils.functions import CurrencyConverter
import os


class AnalysisWidget(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		
		# Создаем выпадающий список для выбора метода анализа
		self.analysis_method_combo = QComboBox(self)
		self.analysis_method_combo.addItems(['Кластерный анализ', 'Анализ месячных затрат', 'Другой анализ'])
		
		# Кнопка для передачи данных на анализ
		self.analyze_button = QPushButton('Передать данные на анализ', self)
		self.analyze_button.clicked.connect(self.send_data_to_analysis)
		
		# Основной макет
		layout = QVBoxLayout(self)
		layout.addWidget(self.analysis_method_combo)
		layout.addWidget(self.analyze_button)
		self.setLayout(layout)
	
	def send_data_to_analysis(self):
		selected_method = self.analysis_method_combo.currentText()
		
		if selected_method == 'Кластерный анализ':
			self.perform_cluster_analysis()
		elif selected_method == 'Анализ месячных затрат':
			self.perform_monthly_expense_analysis()
		elif selected_method == 'Другой анализ':
			self.perform_other_analysis()
		else:
			print("Метод анализа не выбран.")
	
	def perform_cluster_analysis(self):
		print("Выполняем кластерный анализ.")
	
	#  код для кластерного анализа здесь...
	
	def perform_monthly_expense_analysis(self):
		print("Выполняем анализ месячных затрат.")
	
	#  код для анализа месячных затрат здесь...
	
	def perform_other_analysis(self):
		print("Выполняем другой анализ.")
	#  код для другого анализа здесь...


def calculate_herfind_hirshman(data_kp, data_contract):
	print("Запускается метод Хирфендаля-Хиршмана")
	
	merged_df = pd.merge(data_kp, data_contract, on='lot_number', suffixes=('','_contract'))
	# Переведем стоимости в единую валюту EUR и добавим новый столбец total_contract_amount_eur
	converter = CurrencyConverter()
	
	# Конвертируем и сохраняем только нужный столбец
	converted_df = converter.convert_column(
		df=merged_df,
		amount_column='total_contract_amount',
		currency_column='contract_currency'
	)
	merged_df['total_contract_amount_eur'] = converted_df['total_contract_amount']
	
	# Группировка по дисциплине и поставщику
	supplier_stats = (
		merged_df.groupby(['discipline', 'counterparty_name'])['total_contract_amount_eur']
		.sum()
		.reset_index()
	)
	# Общий объем закупок по дисциплине
	total_per_discipline = (
		merged_df.groupby('discipline')['total_contract_amount_eur']
		.sum()
		.reset_index()
		.rename(columns={'total_contract_amount_eur': 'total_discipline_amount'})
	)
	# Добавление общего объема к метрике поставщиков
	supplier_stats = supplier_stats.merge(total_per_discipline, on='discipline')
	
	# Расчет доли поставщика
	supplier_stats['share'] = (
			supplier_stats['total_contract_amount_eur'] / supplier_stats['total_discipline_amount'] * 100
	)
	
	# Вычисление HHI
	hhi = supplier_stats.groupby('discipline')['share'].apply(lambda x: (x ** 2).sum()).reset_index()
	hhi.columns = ['discipline', 'hhi_index']
	
	return supplier_stats, hhi


def find_alternative_suppliers(major_suppliers, data_kp):
	# Создаем директорию для результатов
	output_dir = r"D:\Analysis-Results\hirshman_results"
	os.makedirs(output_dir, exist_ok=True)
	
	# Подготовка объекта для конвертации валют
	converter = CurrencyConverter()
	
	# Результаты анализа
	results = []
	
	# Проходим по каждой дисциплине в major_suppliers
	for discipline in major_suppliers['discipline'].unique():
		# Текущие поставщики для дисциплины
		discipline_suppliers = major_suppliers[major_suppliers['discipline'] == discipline]
		
		# Товары, относящиеся к дисциплине, в data_kp
		discipline_goods = data_kp[data_kp['discipline'] == discipline]
		
		# Если товаров для дисциплины нет, пропускаем
		if discipline_goods.empty:
			continue
		
		# Конвертируем цены за единицу товара в EUR
		discipline_goods = converter.convert_column(
			discipline_goods,
			amount_column='unit_price',
			currency_column='currency',
			result_column='unit_price_eur'
		)
		
		# Анализ товаров и поиск альтернатив
		for good_name in discipline_goods['good_name'].unique():
			# Текущие поставщики для конкретного товара
			current_goods = discipline_goods[discipline_goods['good_name'] == good_name]
			current_suppliers = discipline_suppliers[
				discipline_suppliers['counterparty_name'].isin(current_goods['winner_name'])
			]
			
			# Проверка на пустой current_suppliers
			if current_suppliers.empty:
				continue
			
			# Альтернативные поставщики для товара
			alternative_suppliers = discipline_goods[
				(discipline_goods['good_name'] == good_name) &
				(~discipline_goods['winner_name'].isin(current_suppliers['counterparty_name']))
				]
			
			# Если альтернативные поставщики найдены, сравниваем цены
			if not alternative_suppliers.empty:
				alt_supplier_avg_price = alternative_suppliers['unit_price_eur'].mean()
				current_avg_price = current_goods['unit_price_eur'].mean()
				
				# Формируем рекомендацию
				results.append({
					'discipline': discipline,
					'good_name': good_name,
					'current_suppliers': list(current_suppliers['counterparty_name'].unique()),
					'current_avg_price': current_avg_price,
					'alt_supplier_avg_price': alt_supplier_avg_price,
					'alt_suppliers': list(alternative_suppliers['winner_name'].unique())
				})
	
	# Сохраняем результаты в Excel
	results_df = pd.DataFrame(results)
	results_path = os.path.join(output_dir, "alternative_suppliers_unit_price.xlsx")
	results_df.to_excel(results_path, index=False)
	
	print(f"Анализ альтернативных поставщиков завершен. Результаты сохранены в: {results_path}")
	return results_path
