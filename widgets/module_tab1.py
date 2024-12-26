from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QWidget, QGridLayout, QComboBox, QLabel,
                             QPushButton, QTableView, QFrame, QMessageBox)
import pandas as pd
from dateutil.relativedelta import relativedelta
import sqlite3
from utils.config import SQL_PATH
from utils.PandasModel_previous import PandasModel

class Tab1Widget(QWidget):
	# Определяем сигнал, который будет испускаться при изменении фильтрованных данных
	filtered_data_changed = pyqtSignal(pd.DataFrame)
	
	# конструктор  класса Tab1Widget
	def __init__(self, data_df):
		super().__init__()
		self.data_df = data_df  # Сохраняем исходный DataFrame
		self.filtered_df = data_df  # Изначально filtered_df — это весь DataFrame
		self.range_dict = fill_combobox()  # Вызываем fill_combobox один раз и сохраняем результат
		self.init_ui()  # Инициализация пользовательского интерфейса
	
	def init_ui(self):
		# Создаем макет и виджеты
		self.layout = QGridLayout(self)
		
		# Создаем первый фрейм
		frame1 = QFrame()
		frame1.setFrameShape(QFrame.Box)
		frame1.setFrameShadow(QFrame.Raised)
		self.layout.addWidget(frame1, 0, 0)
		frame1_layout = QGridLayout(frame1)
		
		# Создаем QComboBox и заполняем его
		self.combo_box = QComboBox()
		self.combo_box.addItems(self.range_dict.keys())
		self.combo_box.currentIndexChanged.connect(self.on_combobox_changed)
		
		label = QLabel("Выбираем диапазон загрузки Лотов")
		button = QPushButton("Выберите диапазон из предложенного списка")
		button.setFixedSize(400, 50)
		button.setStyleSheet("""
            QPushButton {
                background-color: rgb(255,153,0);
                color: blue;
                border: none;
                border-radius: 15px;
                padding: 10px 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: rgb(255,185,80);
            }
        """)
		
		frame1_layout.addWidget(label, 0, 0)
		frame1_layout.addWidget(self.combo_box, 0, 1)
		frame1_layout.addWidget(button, 0, 2)
		
		# Создаем QTableView для отображения данных
		self.table_widget = QTableView()
		self.layout.addWidget(self.table_widget, 1, 0, 1, 3)
		
		# Создаем второй фрейм
		frame2 = QFrame()
		frame2.setFrameShape(QFrame.Box)
		self.layout.addWidget(frame2, 2, 0, 1, 3)
		frame2_layout = QGridLayout(frame2)
		sec_frame_label = QLabel("It's a second Frame")
		frame2_layout.addWidget(sec_frame_label, 0, 0)
	
	def on_combobox_changed(self, index):
		# Обработка изменения выбора в combo_box
		selected_name = self.combo_box.currentText()
		start_delta, end_delta = self.range_dict.get(selected_name)
		self.filtered_df = self.filter_data_by_range(self.data_df, start_delta, end_delta)
		self.display_data(self.filtered_df)
		self.filtered_data_changed.emit(self.filtered_df)  # Испускаем сигнал с новыми данными
		print("Updating Tab2 data with filtered DataFrame:", self.filtered_df.head())
	
	def filter_data_by_range(self, data_df, start_delta, end_delta): # диапазон дат и фильтрация данных
		now = pd.to_datetime('now')
		
		try:
			if 'month' in start_delta:
				num_months = int(start_delta.split()[0])
				start_date = now + relativedelta(months=num_months)
			else:
				start_date = now
			
			end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
		except Exception as e:
			print(f"Ошибка при вычислении начальной и конечной даты: {e}")
			return pd.DataFrame()
		
		data_df['close_date'] = pd.to_datetime(data_df['close_date'], errors='coerce')
		data_df = data_df.dropna(subset=['close_date'])
		
		if start_date > end_date:
			QMessageBox.warning(self, "Предупреждение",
			                    "Начальная дата позже конечной даты. Проверьте корректность значений")
			return pd.DataFrame()
		
		filtered_df = data_df[(data_df['close_date'] >= start_date) & (data_df['close_date'] <= end_date)]
		return filtered_df
	
	def display_data(self, df):
		if df.empty:
			QMessageBox.warning(self, "Ошибка", "DataFrame пустой, нечего отображать.")
			return
		model = PandasModel(df)
		self.table_widget.setModel(model)
		self.table_widget.setSortingEnabled(True)
		"""DataFrame отображен в QTableView"""


def fill_combobox():
	with sqlite3.connect(SQL_PATH) as conn:
		cursor = conn.cursor()
		cursor.execute("SELECT name, start_delta, end_delta FROM date_ranges")
		ranges = cursor.fetchall()
	
	range_dict = {name: (start_delta, end_delta) for name, start_delta, end_delta in ranges}
	return range_dict
