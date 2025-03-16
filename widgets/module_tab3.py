"""
	Модуль загрузки и подготовки к обработке Контрактов
"""

from PyQt5.QtCore import pyqtSignal, QDate
from PyQt5.QtWidgets import (QWidget, QGridLayout, QDateEdit, QLabel, QPushButton,
                             QTableView, QFrame, QMessageBox)
import pandas as pd
from utils.config import SQL_PATH
import sqlite3
from utils.date_range_selector_new import DateRangeSelector  # Импортируем класс из отдельного модуля

from utils.functions import clean_contract_data
from utils.PandasModel_previous import PandasModel

class Tab3Widget(QWidget):
	# Определяем сигнал, который будет испускаться при изменении фильтрованных данных
	filtered_contracts_changed = pyqtSignal(pd.DataFrame)
	
	def __init__(self):
		super().__init__()
		self.contract_df = pd.DataFrame()
		self.init_ui() # инициализация пользовательского интерфейса
		
	def init_ui(self):
		# создаем макет и виджеты
		self.layout = QGridLayout(self)
		
		# Выбор диапазона дат загрузки данных
		self.start_date_edit = QDateEdit(self)
		self.start_date_edit.setCalendarPopup(True)
		self.start_date_edit.setDate(QDate.currentDate().addMonths(-1))
		
		self.end_date_edit = QDateEdit(self)
		self.end_date_edit.setCalendarPopup(True)
		self.end_date_edit.setDate(QDate.currentDate())
		
		self.layout.addWidget(QLabel("Начальная дата:"), 0, 0)
		self.layout.addWidget(self.start_date_edit, 0, 1)
		self.layout.addWidget(QLabel("Конечная дата:"), 1, 0)
		self.layout.addWidget(self.end_date_edit, 1, 1)
		
		# Кнопка применения диапазона дат
		self.apply_button = QPushButton("Выбрать диапазон дат загрузки Контрактов")
		self.apply_button.clicked.connect(self.apply_date_filter)
		self.layout.addWidget(self.apply_button, 2, 0, 1, 2)
		
		# Создаем QTableView для отображения данных
		self.table_widget = QTableView()
		self.layout.addWidget(self.table_widget, 3, 0, 1, 3)
		
		# Второй фрейм с информацией о данных
		frame2 = QFrame()
		frame2.setFrameShape(QFrame.Box)
		self.layout.addWidget(frame2, 4, 0, 1, 3)
		
	def apply_date_filter(self):
		# получаем выбранные даты
		start_date = self.start_date_edit.date().toPyDate()
		end_date = self.end_date_edit.date().toPyDate()
		
		# Преобразуем даты в datetime64[ns]
		start_date = pd.to_datetime(start_date)
		end_date = pd.to_datetime(end_date)
		
		# Проверяем корректность диапазона дат
		if start_date > end_date:
			QMessageBox.warning(self, "Предупреждение",
			                    "Начальная дата позже конечной даты. Проверьте корректность значений")
			return
		
		# Загружаем данные из warehouses_remnants по выбранным датам
		db_path = SQL_PATH
		conn = sqlite3.connect(db_path)
		query = f"""
				SELECT * FROM data_contract
				WHERE DATE(contract_signing_date) BETWEEN DATE('{start_date}') AND DATE('{end_date}');
					"""
		# Загрузка данных в датафрейм
		self.contract_df = pd.read_sql_query(query, conn)
		# закрыть соединение с базой данных
		conn.close()
		
		self.contract_df = clean_contract_data(self.contract_df)  # очистка данных полученного df
		
		# Испускаем сигнал с данными
		self.filtered_contracts_changed.emit(self.contract_df)
		
		self.display_data(self.contract_df)
		
	def display_data(self, contract_df):
		if contract_df.empty:
			QMessageBox.warning(self, "Ошибка", "DataFrame пустой, нечего отображать")
			return
		try:
			model = PandasModel(contract_df)
			self.table_widget.setModel(model)
			self.table_widget.setSortingEnabled(True)
		except Exception as e:
			QMessageBox.critical(self, "Ошибка", f"Не удалось отобразить данные: {str(e)}")
