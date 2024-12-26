from styles import set_light_theme, set_fonts, load_stylesheet

from utils.data_model import DataModel
from utils.clean_datas import clean_database
from utils.functions import load_data_from_sql, cleanDataDF, load_data_contract_from_sql
from utils.visualizer import KPIVisualizer
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QInputDialog, QLabel,
                             QStatusBar, QProgressBar, QTabWidget, QVBoxLayout, QMessageBox, QWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt5.QtWidgets import QToolTip
from PyQt5.QtGui import QFont, QCursor

from widgets.module_tab1 import Tab1Widget
# from widgets.module_tab2 import Tab2
from widgets.module_universe import TabWidget
from widgets.module_tab3 import Tab3Widget
# from widgets.module_tab4 import Tab4
import json
import pandas as pd


# Загрузка подсказок из JSON-файла
def load_menu_hints():
	with open('menu_hints.json', 'r', encoding='utf-8') as file:
		return json.load(file)


class DataCleanerThread(QThread):
	progress_changed = pyqtSignal(int)
	
	def run(self):
		"""Выполнение очистки данных в фоновом потоке."""
		self.progress_changed.emit(0)
		clean_database()
		self.progress_changed.emit(100)


def clicked_connect(self):
	"""эта функция вызывает метод open_file из класса Data_model модуля data_model.py"""
	DataModel(self).open_file_dialog()


class MyTabWidget(QWidget):
	def __init__(self, data_df, contract_df, contracts_count, future_dates_count, invalid_year_count,
	             missing_unit_price_count,
	             negative_price_count, invalid_signing_date_count, missing_executor_dak_count):
		super().__init__()
		self.df_kpi_normalized = None
		self.data_df = data_df
		self.contract_df = contract_df
		self.notebook = QTabWidget()
		self.setup_tabs(contracts_count, future_dates_count, invalid_year_count, missing_unit_price_count,
		                negative_price_count, invalid_signing_date_count, missing_executor_dak_count)
		layout = QVBoxLayout(self)
		layout.addWidget(self.notebook)
		self.setLayout(layout)
		
		# Инициализация QLabel для отображения подсказок
		self.tooltip_label = QLabel(self)
		self.tooltip_label.setStyleSheet(
			"background-color: yellow; color: black; padding: 5px; border: 1px solid black;")
		self.tooltip_label.hide()  # Скрываем по умолчанию
	
	def showTooltip(self, text, x=20, y=20):
		try:
			# Устанавливаем текст подсказки и показываем ее в фиксированной позиции для отладки
			self.tooltip_label.setText(text)
			cursor_pos = QCursor.pos()
			self.tooltip_label.move(self.mapFromGlobal(cursor_pos + QPoint(x, y)))
			self.tooltip_label.adjustSize()
			self.tooltip_label.show()
		except Exception as error:
			print("Ошибка при отображении подсказки:", error)
	
	def hideTooltip(self):
		# Скрываем виджет подсказки
		self.tooltip_label.hide()
	
	def setup_tabs(self, contracts_count, future_dates_count, invalid_year_count, missing_unit_price_count,
	               negative_price_count, invalid_signing_date_count, missing_executor_dak_count):
		tab1 = Tab1Widget(self.data_df)
		tab2 = TabWidget(
			['lot_number', 'project_name', 'discipline', 'actor_name', 'winner_name', 'currency', 'good_name'])
		tab3 = Tab3Widget(self.contract_df, contracts_count, future_dates_count, invalid_year_count,
		                  missing_unit_price_count, negative_price_count, invalid_signing_date_count,
		                  missing_executor_dak_count)
		tab4 = TabWidget(
			['lot_number', 'discipline', 'contract_name', 'executor_dak', 'counterparty_name', 'product_name',
			 'contract_currency'])
		
		self.notebook.addTab(tab1, 'Данные по Лотам')
		self.notebook.addTab(tab2, 'Параметры загруженных Лотов')
		self.notebook.addTab(tab3, 'Данные по Контрактам')
		self.notebook.addTab(tab4, 'Параметры загруженных Контрактов')
		
		# подключаем сигнал для взаимодействия между вкладками
		tab1.filtered_data_changed.connect(tab2.update_data)
		tab1.filtered_data_changed.connect(tab3.update_contract_data)
		# Подключение сигнала от Tab3 к Tab4 для передачи отфильтрованных контрактов
		tab3.filtered_contracts_changed.connect(tab4.on_filtered_contracts_received)
		print("MyTabWidget: Вкладки успешно созданы")  # Отладочный принт


class Window(QMainWindow):
	def __init__(self):
		super().__init__()
		self.data_df = None
		self.contract_df = None
		self.filtered_df = None
		
		# Загрузка подсказок
		self.menu_hints = load_menu_hints()
		
		# Загрузка всех данных из таблицы data_kp базы данных
		self.data_df = load_data_from_sql()  # Загружаем весь DataFrame при старте
		self.data_df = cleanDataDF(self.data_df)  # очистка данных полученного df
		
		# Загрузка всех данных из таблицы data_contract базы данных и получение статистики
		(self.contract_df,
		 self.contracts_count,
		 self.future_dates_count,
		 self.invalid_year_count,
		 self.missing_unit_price_count,
		 self.negative_price_count,
		 self.invalid_signing_date_count,
		 self.missing_executor_dak_count) = load_data_contract_from_sql()
		
		# Используем QTimer.singleShot, чтобы показать сообщения после загрузки GUI
		QTimer.singleShot(0, lambda: QMessageBox.information(self, "Инфо", "Загружены все Лоты и все Контракты"))
		
		# Создание и установка вкладок
		self.tab_widget = MyTabWidget(
			self.data_df,
			self.contract_df,
			self.contracts_count,
			self.future_dates_count,
			self.invalid_year_count,
			self.missing_unit_price_count,
			self.negative_price_count,
			self.invalid_signing_date_count,
			self.missing_executor_dak_count
		)
		self.setCentralWidget(self.tab_widget)
		
		# Подключение сигнала для обновления данных между вкладками
		tab1_widget = self.tab_widget.notebook.widget(0)  # Получаем первый виджет вкладки
		if isinstance(tab1_widget, Tab1Widget):
			tab1_widget.filtered_data_changed.connect(self.update_tab2_data)
		
		# Подключение сигнала для получения отфильтрованных данных
		tab2_widget = self.tab_widget.notebook.widget(1)  # Получаем второй виджет вкладки (Tab2)
		if isinstance(tab2_widget, TabWidget):
			tab2_widget.data_ready_for_analysis.connect(self.set_filtered_data)
		
		# Подключение сигнала для обновления данных между вкладками
		tab3_widget = self.tab_widget.notebook.widget(2)
		if isinstance(tab3_widget, TabWidget):
			tab3_widget.data_ready_for_analysis.connect(self.update_tab3_data)
		
		tab4_widget = self.tab_widget.notebook.widget(3)
		if isinstance(tab4_widget, TabWidget):
			tab4_widget.data_ready_for_analysis.connect(self.set_filtered_data)
		
		# Настройка главного окна
		self.setFont(QFont("Arial", 12))
		self.setWindowTitle('Анализ закупочных процессов')
		self.resize(1200, 600)
		self.setFont(QFont("Arial", 11))
		
		# Создание меню и действий
		self._createActions()
		self._createMenuBar()
		self._connectActions()
		
		# Создание статусной строки и прогрессбара
		self.status_bar = QStatusBar(self)
		self.progress_bar = QProgressBar(self)
		self.progress_bar.setMinimum(0)
		self.progress_bar.setMaximum(100)
		self.progress_bar.setValue(0)
		self.status_bar.addPermanentWidget(self.progress_bar)
		self.setStatusBar(self.status_bar)
	
	def update_tab2_data(self, filtered_df):
		self.tab_widget.notebook.widget(1).update_data(filtered_df)
	
	def update_tab3_data(self, updated_contract_df):
		# Логика обновления данных на вкладке 3
		self.tab_widget.notebook.widget(2).update_data(updated_contract_df)
		print("Данные для вкладки 3 обновлены")
	
	def _createMenuBar(self):
		menuBar = self.menuBar()
		
		# Меню Файл
		fileMenu = menuBar.addMenu("Ввод основной информации")
		fileMenu.addAction(self.ContrAction)
		fileMenu.addAction(self.CleanDatas)
		fileMenu.addAction(self.GetBasData)
		fileMenu.addSeparator()
		fileMenu.addAction(self.ExitAction)
		
		# Меню Анализ по Лотам
		analysisMenu = menuBar.addMenu("Анализ данных по Лотам")
		analysisMenu.addAction(self.analyzeMonthlyExpensesAction)
		analysisMenu.addAction(self.analyzeTopSuppliersAction)
		analysisMenu.addAction(self.analyzeClasterAction)
		analysisMenu.addAction(self.suppliersfriquencyAction)
		analysisMenu.addAction(self.networkanalyseAction)
		analysisMenu.addAction(self.analyzeKPIAction)
		analysisMenu.addAction(self.efficiency_analyses_action)
		analysisMenu.addAction(self.suppliers_by_unit_price_action)
		analysisMenu.addAction(self.find_cross_discipline_lotsAction)
		
		# Меню Анализ по Контрактам
		analysisMenuContract = menuBar.addMenu('Анализ данных по Контрактам')
		analysisMenuContract.addAction(self.analyzeNoneEquilSums)
		analysisMenuContract.addAction(self.trend_analyses_action)
		analysisMenuContract.addAction(self.prophet_arima_action)
		analysisMenuContract.addAction(self.contracts_less_dates_action)
	
	def setActionTooltip(self, action, group, hint_key, x=20, y=20):
		hint_text = self.menu_hints.get(group, {}).get(hint_key, "Нет инструкции для этого пункта")
		action.hovered.connect(lambda: self.tab_widget.showTooltip(hint_text))
		action.triggered.connect(self.tab_widget.hideTooltip)
	
	def leaveEvent(self, event):
		self.tab_widget.hideTooltip()
		super().leaveEvent(event)
	
	def _createActions(self):
		# Действия для меню Файл
		self.ContrAction = QAction("Загрузить данные из Отчетов", self)
		self.CleanDatas = QAction("Очистить данные в БД", self)
		self.GetBasData = QAction("Получить основные данные", self)
		self.ExitAction = QAction("Выход", self)
		
		self.statusBar().showMessage('Все ОК')
		
		# Действия для меню Анализ данных по Лотам
		self.analyzeMonthlyExpensesAction = QAction("Анализ месячных затрат", self)
		self.setActionTooltip(self.analyzeMonthlyExpensesAction, "Анализ данных по Лотам", "menu_item_1", x=20, y=20)
		
		self.analyzeTopSuppliersAction = QAction("Анализ топ-10 поставщиков", self)
		self.setActionTooltip(self.analyzeTopSuppliersAction, "Анализ данных по Лотам", "menu_item_2", x=20, y=20)
		
		self.analyzeClasterAction = QAction("Кластерный анализ", self)
		self.setActionTooltip(self.analyzeClasterAction, "Анализ данных по Лотам", "menu_item_3", x=20, y=20)
		
		self.suppliersfriquencyAction = QAction("Анализ частоты поставщиков", self)
		self.setActionTooltip(self.suppliersfriquencyAction, "Анализ данных по Лотам", "menu_item_4", x=20, y=20)
		
		self.networkanalyseAction = QAction("Сетевой анализ проектов", self)
		self.setActionTooltip(self.networkanalyseAction, "Анализ данных по Лотам", "menu_item_5", x=20, y=20)
		
		self.analyzeKPIAction = QAction("Анализ KPI", self)
		self.setActionTooltip(self.analyzeKPIAction, "Анализ данных по Лотам", "menu_item_6", x=20, y=20)
		
		self.efficiency_analyses_action = QAction("Анализ эффективности исполнителей и поиск аномалий", self)
		self.setActionTooltip(self.efficiency_analyses_action, "Анализ данных по Лотам", "menu_item_7",
		                      x=20, y=20)
		self.suppliers_by_unit_price_action = QAction("Ранжирование Поставщиков по цене за единицу товара", self)
		self.setActionTooltip(self.suppliers_by_unit_price_action, "Анализ данных по Лотам", "menu_item_8",
		                      x=20, y=20)
		self.find_cross_discipline_lotsAction = QAction("Поиск и анализ лотов общих для разных дисциплин", self)
		self.setActionTooltip(self.find_cross_discipline_lotsAction, "Анализ данных по Лотам", "menu_item_9",
		                      x=20, y=20)
		
		# Действия для меню Анализ данных по Контрактам
		self.analyzeNoneEquilSums = QAction('Поиск и анализ несоответствий в суммах Лотов и Контрактов', self)
		self.setActionTooltip(self.analyzeNoneEquilSums, "Анализ данных по Контрактам", "menu_item_1", x=450, y=20)
		
		self.trend_analyses_action = QAction('Тренд - анализ', self)
		self.setActionTooltip(self.trend_analyses_action, "Анализ данных по Контрактам", "menu_item_2", x=450, y=20)
		
		self.prophet_arima_action = QAction('Моделирование и прогнозирование', self)
		self.setActionTooltip(self.prophet_arima_action, "Анализ данных по Контрактам", "menu_item_3", x=450, y=20)
		
		self.contracts_less_dates_action = QAction('Поиск и анализ контрактов с инвалидными датами ', self)
		self.setActionTooltip(self.contracts_less_dates_action, "Анализ данных по Контрактам", "menu_item_4", x=450,
		                      y=20)
	
	def _connectActions(self):
		# Подключение сигналов к действиям
		self.ContrAction.triggered.connect(self.load_sql_data)
		self.CleanDatas.triggered.connect(self.run_clean_data)
		self.ExitAction.triggered.connect(self.close)
		
		# Подключение сигналов к методам Анализа данных по Лотам
		self.analyzeMonthlyExpensesAction.triggered.connect(self.run_analyze_monthly_expenses)
		self.analyzeTopSuppliersAction.triggered.connect(self.run_analyze_top_suppliers)
		self.analyzeClasterAction.triggered.connect(self.run_ClusterAnalyze)
		self.suppliersfriquencyAction.triggered.connect(self.run_analyze_supplier_friquency)
		self.networkanalyseAction.triggered.connect(self.run_network_analysis)
		self.analyzeKPIAction.triggered.connect(self.run_kpi_analysis)
		self.efficiency_analyses_action.triggered.connect(self.run_efficiency_analyses)
		self.suppliers_by_unit_price_action.triggered.connect(self.run_analyze_by_unit_price)
		self.find_cross_discipline_lotsAction.triggered.connect(self.run_find_cross_discipline_lots)
		
		# Подключение сигналов к методам Анализа данных по Контрактам
		self.analyzeNoneEquilSums.triggered.connect(self.run_analyzeNonEquilSums)
		self.trend_analyses_action.triggered.connect(self.run_trend_analyses)
		self.prophet_arima_action.triggered.connect(self.run_prophet_and_arima)
		self.contracts_less_dates_action.triggered.connect(self.run_contracts_less_dates)
	
	def set_filtered_data(self, df):
		"""Устанавливает отфильтрованный DataFrame для анализа."""
		self.filtered_df = df
		print("Обновленные данные для анализа:")
		print(self.filtered_df.head())  # Вывод первых строк DataFrame для проверки
		QMessageBox.information(self, "Информация", "Данные для анализа успешно обновлены.")
	
	def run_clean_data(self):
		# self.progress_bar.setValue(0)
		clean_database()
	
	# self.progress_bar.setValue(100)
	
	def run_kpi_analysis(self):
		"""Запуск анализа KPI с использованием отфильтрованных данных."""
		if self.filtered_df is not None:
			from models_analyses.MyLotAnalyzeKPI import LotAnalyzeKPI
			
			# Передаем отфильтрованные данные в KPI анализатор
			kpi_analyzer = LotAnalyzeKPI(self.filtered_df)
			self.df_kpi_normalized = kpi_analyzer.calculate_kpi(self.filtered_df)
			print("Содержимое self.df_kpi_normalized после вычисления:")
			print(self.df_kpi_normalized.describe())
			
			# Визуализация KPI
			self.visualize_kpi()
			
			QMessageBox.information(self, "KPI Анализ", "KPI анализ успешно завершен.")
		else:
			QMessageBox.warning(self, "Ошибка", "Нет отфильтрованных данных для анализа KPI.")
	
	def visualize_kpi(self):
		"""Вызов визуализации KPI."""
		if hasattr(self, 'df_kpi_normalized') and self.df_kpi_normalized is not None:
			visualizer = KPIVisualizer(self.df_kpi_normalized)
			
			# Создаем диалог для выбора типа визуализации
			dialog = QMessageBox(self)
			dialog.setWindowTitle("Выбор типа визуализации")
			dialog.setText("Выберите тип визуализации KPI:")
			bar_btn = dialog.addButton("Бар-чарт", QMessageBox.ActionRole)
			pie_btn = dialog.addButton("Круговая диаграмма", QMessageBox.ActionRole)
			heatmap_btn = dialog.addButton("Тепловая карта", QMessageBox.ActionRole)
			line_btn = dialog.addButton("Линейный график", QMessageBox.ActionRole)
			dialog.exec_()
			
			clicked_button = dialog.clickedButton()
			
			if clicked_button == bar_btn:
				visualizer.plot_bar_chart()
			elif clicked_button == pie_btn:
				visualizer.plot_pie_chart()
			elif clicked_button == heatmap_btn:
				visualizer.plot_heatmap()
			elif clicked_button == line_btn:
				visualizer.plot_line_chart()
		else:
			QMessageBox.warning(self, "Ошибка", "Нет данных KPI для визуализации.")
	
	def run_analyze_monthly_expenses(self):
		# метод для анализа месячных затрат
		if self.filtered_df is not None:
			print("Данные для анализа (месячные затраты):")
			print(self.filtered_df.head())  # Вывод первых строк для проверки
			print(self.filtered_df['project_name'])
			# Используем минимальную и максимальную даты из отфильтрованных данных
			start_date = self.filtered_df['close_date'].min()
			end_date = self.filtered_df['close_date'].max()
			from models_analyses.analysis import analyze_monthly_expenses
			analyze_monthly_expenses(self.filtered_df, start_date, end_date)
		# Ваша логика для анализа данных
		else:
			QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")
	
	def run_analyze_top_suppliers(self):
		#  метод анализа поставщиков с высокими и низкими ценами за единицу товара
		if self.filtered_df is not None:
			"""Используем минимальную и максимальную даты из отфильтрованных данных"""
			start_date = self.filtered_df['close_date'].min()
			end_date = self.filtered_df['close_date'].max()
			from models_analyses.analysis import analyze_top_suppliers
			analyze_top_suppliers(self, self.filtered_df, start_date, end_date)
		# Логика для анализа данных
		else:
			QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")
	
	def run_ClusterAnalyze(self):
		# Метод для классификации исполнителей с обучением методом SeedKMeans
		if self.filtered_df is not None:
			from models_analyses.MyLotAnalyzeKPI import LotAnalyzeKPI
			from models_analyses.SeedKMeans_clustering import SeedKMeansClustering, export_to_excel
			import logging
			import os
			
			output_dir = 'D:\Analysis-Results\clussification_analysis'
			os.makedirs(output_dir, exist_ok=True)
			
			# Создаем объект KPI-анализатора
			kpi_analyzer = LotAnalyzeKPI(self.filtered_df)
			
			# Создаем объект для кластеризации, передавая KPI-анализатор
			clustering_module = SeedKMeansClustering(kpi_analyzer)
			df_clusters, kmeans_model = clustering_module.perform_clustering()
			
			if df_clusters is not None:
				# Сохранение гистограммы
				histogram_path = os.path.join(output_dir, 'cluster_distribution.png')
				clustering_module.plot_cluster_distribution(df_clusters, histogram_path)
				
				# Сохранение данных в Excel
				excel_path = os.path.join(output_dir, 'cluster_analysis_report.xlsx')
				export_to_excel(df_clusters, excel_path)
				
				logging.info(f'Кластерный анализ завершен. Результаты сохранены в {output_dir}')
			else:
				logging.error('Кластерный анализ завершился ошибкой.')
	
	def run_analyze_supplier_friquency(self):
		# Метод для анализа частоты выбора поставщиков
		if self.filtered_df is not None:
			from models_analyses.analysis import analyze_supplier_frequency
			analyze_supplier_frequency(self.filtered_df)
		else:
			QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")
	
	def run_network_analysis(self):
		# Метод для сетевого анализа
		if self.filtered_df is not None:
			print("Запуск сетевого анализа")
			from models_analyses.analysis import network_analysis
			network_analysis(self, self.filtered_df)
		# print(result)
		else:
			QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")
	
	def load_sql_data(self):
		print("Загрузка данных в базу SQL...")
		DataModel(self).open_file_dialog()
	
	# Логика работы с Анализом данных по Контрактам
	
	def run_analyzeNonEquilSums(self):
		# Метод для поиска несоответствий в суммах Лотов и Контрактов
		print("Запуск поиска несоответствий сумм")
		from models_analyses.analyze_contracts import analyzeNonEquilSums
		analyzeNonEquilSums(self.data_df, self.contract_df)
	
	def run_trend_analyses(self):
		from models_analyses.analyze_contracts import show_interactive_trend
		print('Пересылаем этот датафрейм')
		print(self.filtered_df)
		# analyze_trend_by_discipline(self.filtered_df, output_folder=r"D:\contracts_trend_analyses")
		show_interactive_trend(self.filtered_df, output_folder=r"D:\contracts_trend_analyses")
	
	# построение множественной регресии и корреляционный анализ
	def run_prophet_and_arima(self):
		from models_analyses.regression_analyses import regresion_analyses
		# в метод регрессионного анализа отправляем отфильтрованный filtered_df и общую базу контрактов
		regresion_analyses(self.filtered_df, self.contract_df)
	
	# анализ контрактов без соответствующих лотов
	def run_contracts_less_dates(self):
		from models_analyses.contracts_without_lots import check_contracts_less_dates
		# метод поиска контрактов без лотов
		check_contracts_less_dates(self.data_df)
	
	def run_efficiency_analyses(self):
		from models_analyses.efficiency_analyses import main_method
		main_method(self.filtered_df, self.data_df)
	
	def run_analyze_by_unit_price(self):
		from models_analyses.efficiency_analyses import analyze_suppliers_by_unit_price
		analyze_suppliers_by_unit_price(self, self.filtered_df, self.data_df)
	
	def run_find_cross_discipline_lots(self):
		import os
		# Шаг 1. Находим общих поставщиков для дисциплин
		from models_analyses.analysis import find_common_suppliers_between_disciplines
		common_suppliers_df = find_common_suppliers_between_disciplines(self.filtered_df)
		if not common_suppliers_df.empty:
			print(common_suppliers_df.columns)
			from models_analyses.analysis import compare_materials_and_prices
			# Шаг 2. Сравниваем цены за единицу продукции
			comparison_results = compare_materials_and_prices(self.filtered_df, common_suppliers_df)
			if not comparison_results.empty:
				print("Сравнение материалов и цен между дисциплинами:")
				print(comparison_results.columns)
				
				# Шаг 3. Визуализация
				from utils.vizualization_tools import visualize_price_differences, heatmap_common_suppliers
				print('Эти данные отправляются на визуализацию')
				print(comparison_results.columns)
				visualize_price_differences(comparison_results)
				heatmap_common_suppliers(common_suppliers_df)
				# Шаг 4. Статистика
				from models_analyses.analysis import matches_results_stat
				matches_results_stat(comparison_results)
			else:
				print("Нет данных для сравнения материалов и цен.")
		else:
			print("Общие поставщики между дисциплинами не найдены.")




if __name__ == "__main__":
	app = QApplication(sys.argv)
	
	# Устанавливаем шрифты и тему
	# set_dark_theme(app)
	set_fonts(app)
	# Загружаем стили из CSS-файла
	stylesheet = load_stylesheet("styles_black.qss")
	app.setStyleSheet(stylesheet)
	# Установим шрифт для отображения подсказок
	QToolTip.setFont(QFont('SansSerif', 10))
	# Установим стиль приложения
	app.setStyle('Fusion')
	set_light_theme(app)
	
	app.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # Включить поддержку высокого разрешения
	app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # Включить использование DPI картинок
	app.setAttribute(Qt.AA_DisableWindowContextHelpButton, False)  # Активировать tooltips
	
	window = Window()
	window.show()
	sys.exit(app.exec_())
