from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QWidget, QGridLayout, QComboBox, QLabel, QHeaderView,
                             QPushButton, QTableView, QFrame, QMessageBox)
import pandas as pd
from dateutil.relativedelta import relativedelta
import sqlite3
from utils.config import SQL_PATH
from utils.PandasModel_previous import PandasModel

class Tab3Widget(QWidget):
	filtered_contracts_changed = pyqtSignal(pd.DataFrame)  # Сигнал для передачи данных в Tab4
	
	# конструктор  класса Tab3Widget
	def __init__(self, contract_df, contracts_count, future_dates_count, invalid_year_count,
                 missing_unit_price_count, negative_price_count, invalid_signing_date_count, missing_executor_dak_count):
		super().__init__()
		self.contract_df = contract_df  # Сохраняем исходный DataFrame
		self.filtered_contr_df = self.contract_df  # Изначально filtered_contract_df — это весь DataFrame
		self.range_dict = fill_combobox()  # Вызываем fill_combobox один раз и сохраняем результат
		
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
		
	
	def update_contract_data(self):
		pass
	
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
		
		label = QLabel("Выбираем диапазон загрузки Контрактов")
		frame1_layout.addWidget(label, 0, 0)
		frame1_layout.addWidget(self.combo_box, 0, 1)
		
		# Создаем QTableView для отображения данных
		self.table_widget = QTableView()
		self.layout.addWidget(self.table_widget, 1, 0, 1, 3)
		
		# Создаем второй фрейм
		frame2 = QFrame()
		frame2.setFrameShape(QFrame.Box)
		self.layout.addWidget(frame2, 2, 0, 1, 3)
		frame2_layout = QGridLayout(frame2)
		sec_frame_label = QLabel(self.sec_frame_text)
		frame2_layout.addWidget(sec_frame_label, 0, 0)
		print('Первый и второй фреймы созданы')

	# Добавим логику для обновления отображения данных
	def on_combobox_changed(self, index):
		print(self.contract_df) #DataFrame загружается
		# Обработка изменения выбора в combo_box
		selected_name = self.combo_box.currentText()
		print('Выбранное имя ', selected_name)
		start_delta, end_delta = self.range_dict.get(selected_name)
		self.filtered_contr_df = self.filter_contr_by_range(self.contract_df, start_delta, end_delta)
		print('Фильтрованный датафрейм')
		print(self.filtered_contr_df)
		self.display_data(self.filtered_contr_df)
		self.filtered_contracts_changed.emit(self.filtered_contr_df)  # Испускаем сигнал с новыми данными
	
	def filter_contr_by_range(self, contract_df, start_delta, end_delta):  # диапазон дат и фильтрация данных
		now = pd.to_datetime('now')
		"""------------"""
		# Проверка на пустые значения в столбце contract_signing_date
		missing_dates = contract_df[contract_df['contract_signing_date'].isna()]
		
		if not missing_dates.empty:
			print("Есть строки с пропущенными датами:")
			print(missing_dates)
		else:
			print("Все даты корректны")
		
		
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
		
		if start_date > end_date:
			QMessageBox.warning(self, "Предупреждение",
			                    "Начальная дата позже конечной даты. Проверьте корректность значений")
			return pd.DataFrame()
		
		print("Текущая дата:", now)
		print("Дата начала (24 месяца назад):", start_date)
		print("Всего строк до фильтрации:", len(contract_df))
		
		# Найдем строки, где год даты больше 2100 или меньше 1900
		out_of_range_dates = contract_df[
			(contract_df['contract_signing_date'].dt.year > 2100) | (contract_df['contract_signing_date'].dt.year < 1900)]
		
		print(out_of_range_dates)
		
		filtered_contr_df = contract_df[(contract_df['contract_signing_date'] >= start_date) & (contract_df['contract_signing_date'] <= end_date)]
		print('Очищенный', filtered_contr_df)
		return filtered_contr_df
	
	def display_data(self, filtered_contr_df):
		print('df без префикса self ', filtered_contr_df)
		if self.filtered_contr_df.empty:
			QMessageBox.warning(self, "Ошибка", "DataFrame пустой, нечего отображать.")
			return
		try:
			print('Создаем Модель')
			model = PandasModel(filtered_contr_df)
			self.table_widget.setModel(model)
			self.table_widget.setSortingEnabled(True)
			"""DataFrame отображен в QTableView"""
		except Exception as e:
			QMessageBox.critical(self, "Ошибка", f"Не удалось отобразить данные: {str(e)}")

def fill_combobox():
	with sqlite3.connect(SQL_PATH) as conn:
		cursor = conn.cursor()
		cursor.execute("SELECT name, start_delta, end_delta FROM date_ranges")
		ranges = cursor.fetchall()
	
	range_dict = {name: (start_delta, end_delta) for name, start_delta, end_delta in ranges}
	return range_dict

