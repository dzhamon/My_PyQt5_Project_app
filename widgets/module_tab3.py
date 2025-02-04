from PyQt5.QtCore import pyqtSignal, QDate
from PyQt5.QtWidgets import (QWidget, QGridLayout, QDateEdit, QLabel, QPushButton, QTableView, QFrame, QMessageBox)
import pandas as pd
from utils.PandasModel_previous import PandasModel


class Tab3Widget(QWidget):
	filtered_contracts_changed = pyqtSignal(pd.DataFrame)  # Сигнал для передачи данных в Tab4
	
	# конструктор класса Tab3Widget
	def __init__(self, contract_df, contracts_count, future_dates_count, invalid_year_count,
	             missing_unit_price_count, negative_price_count, invalid_signing_date_count,
	             missing_executor_dak_count):
		super().__init__()
		self.contract_df = contract_df  # Сохраняем исходный DataFrame
		self.filtered_contr_df = self.contract_df  # Изначально filtered_contract_df — это весь DataFrame
		
		self.sec_frame_text = (
			f"Количество контрактов в базе данных: {contracts_count}\n"
			f"Количество контрактов с датой подписания в будущем: {future_dates_count}\n"
			f"Количество контрактов с годом подписания < 1900: {invalid_year_count}\n"
			f"Контракты с отсутствующей ценой: {missing_unit_price_count}\n"
			f"Контракты с отрицательной ценой: {negative_price_count}\n"
			f"Контракты с некорректной датой подписания (раньше даты окончания лота): {invalid_signing_date_count}\n"
			f"Количество строк с пропущенными именами исполнителей контрактов: {missing_executor_dak_count}"
		)
		self.init_ui()  # Инициализация пользовательского интерфейса
		
		"""
			Приняв вышенайденные переменные, вывести их в отдельный файл, затем прняться , если это возможно,
			за чистку полученных данных. В противном случае удачить их из загруженного датафрейма
		"""
	
	def update_contract_data(self, filtered_data):
		"""
		Метод для обновления данных в Tab3Widget на основе фильтрованных данных из Tab1Widget.
		"""
		try:
			# Обновляем DataFrame на основе полученных данных
			self.contract_df = filtered_data
			self.filtered_contr_df = filtered_data
			# Обновляем отображение данных
			self.display_data(self.filtered_contr_df)
			print("Данные в Tab3 обновлены.")
		except Exception as e:
			QMessageBox.critical(self, "Ошибка", f"Не удалось обновить данные: {str(e)}")
	
	def init_ui(self):
		# Создаем макет и виджеты
		self.layout = QGridLayout(self)
		
		# Выбор диапазона дат
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
		self.apply_button = QPushButton("Применить диапазон дат")
		self.apply_button.clicked.connect(self.apply_date_filter)
		self.layout.addWidget(self.apply_button, 2, 0, 1, 2)
		
		# QTableView для отображения данных
		self.table_widget = QTableView()
		self.layout.addWidget(self.table_widget, 3, 0, 1, 3)
		
		# Второй фрейм с информацией о данных
		frame2 = QFrame()
		frame2.setFrameShape(QFrame.Box)
		self.layout.addWidget(frame2, 4, 0, 1, 3)
		frame2_layout = QGridLayout(frame2)
		sec_frame_label = QLabel(self.sec_frame_text)
		frame2_layout.addWidget(sec_frame_label, 0, 0)
	
	def apply_date_filter(self):
		# Получаем выбранные даты
		start_date = self.start_date_edit.date().toPyDate()
		end_date = self.end_date_edit.date().toPyDate()
		
		# Проверяем корректность диапазона дат
		if start_date > end_date:
			QMessageBox.warning(self, "Предупреждение",
			                    "Начальная дата позже конечной даты. Проверьте корректность значений")
			return
		
		# Фильтрация данных
		self.filtered_contr_df = self.filter_contr_by_range(self.contract_df, start_date, end_date)
		self.display_data(self.filtered_contr_df)
		self.filtered_contracts_changed.emit(self.filtered_contr_df)  # Испускаем сигнал с новыми данными
	
	def filter_contr_by_range(self, contract_df, start_date, end_date):
		# Фильтрация данных
		contract_df['contract_signing_date'] = pd.to_datetime(contract_df['contract_signing_date'], errors='coerce')
		contract_df = contract_df.dropna(subset=['contract_signing_date'])
		
		filtered_contr_df = contract_df[
			(contract_df['contract_signing_date'] >= pd.to_datetime(start_date)) &
			(contract_df['contract_signing_date'] <= pd.to_datetime(end_date))
			]
		return filtered_contr_df
	
	def display_data(self, filtered_contr_df):
		if filtered_contr_df.empty:
			QMessageBox.warning(self, "Ошибка", "DataFrame пустой, нечего отображать.")
			return
		
		try:
			model = PandasModel(filtered_contr_df)
			self.table_widget.setModel(model)
			self.table_widget.setSortingEnabled(True)
		except Exception as e:
			QMessageBox.critical(self, "Ошибка", f"Не удалось отобразить данные: {str(e)}")
