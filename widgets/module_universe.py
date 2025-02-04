from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QWidget, QListWidget, QMessageBox, QLineEdit,
                             QVBoxLayout, QPushButton, QTableView, QDialog, QAbstractItemView)
from PyQt5.QtCore import pyqtSignal
from utils.PandasModel_previous import PandasModel
import pandas as pd


class TabWidget(QWidget):
	data_ready_for_analysis = pyqtSignal(pd.DataFrame)
	
	def __init__(self, list_names):
		super().__init__()
		
		# создание основного макета
		self.list_names = list_names
		self.layout = QVBoxLayout(self)
		# словарь для хранения QListWidget и их значений
		self.list_widgets = {}
		self.filtered_df = pd.DataFrame() # пустой df для отфильтр значений
		
		# создание макета для списков (горизонтальный макет)
		list_layout = QHBoxLayout()
		# инициализация 6 списков и кнопок "Очистить"
		self.init_list_widgets(list_layout)
		
		# создание кнопки для выполнения запроса
		self.execute_button = QPushButton('Выполнить запрос')
		self.execute_button.clicked.connect(self.show_filtered_data)
		
		# добавление макета списков в основной макет
		self.layout.addLayout(list_layout)
		self.layout.addWidget(self.execute_button)
		self.setLayout(self.layout) # устанавливаем основной макет
		
	def send_data_to_analysis(self):
		"""
			Передает отфильтрованные данные в модуль анализа
		"""
		if hasattr(self, 'filtered_df_to_analyze') and not self.filtered_df_to_analyze.empty:
			self.data_ready_for_analysis.emit(self.filtered_df_to_analyze)
		else:
			QMessageBox.warning(self, "Ошибка", "Нет данных для анализа")
	
	def init_list_widgets(self, layout):
		for name in self.list_names:
			widget_layout = QVBoxLayout()
			label = QLabel(f"Выберите {name}:")
			search_entry = QLineEdit(self)
			search_entry.setPlaceholderText("Начните вводить для поиска...")
			search_entry.show()
			
			list_widget = QListWidget()
			list_widget.setObjectName(name)
			list_widget.setSelectionMode(QListWidget.MultiSelection)
			list_widget.show()
			self.list_widgets[name] = (list_widget, search_entry)
			# привязываем поиск к строке поиска
			search_entry.textChanged.connect(lambda text, lw=list_widget: self.filter_listbox(text, lw))
			
			# кнопка "очистить"
			clear_button = QPushButton('Очистить')
			clear_button.clicked.connect(lambda _, lw=list_widget: self.clear_selection(lw))
			
			# Добавление виджетов в макет
			widget_layout.addWidget(label)
			widget_layout.addWidget(search_entry)
			widget_layout.addWidget(list_widget)
			widget_layout.addWidget(clear_button)
			layout.addLayout(widget_layout)
	
	def filter_listbox(self, text, list_widget):
		"""
			Фильтрует элементы в списке на основе текста, сохраняя полный список.
		"""
		
		# Получаем полный список элементов
		if not hasattr(list_widget, 'full_list'):
			list_widget.full_list = [list_widget.item(row).text() for row in range(list_widget.count())]
		
		# Фильтруем элементы
		filtered_items = [item for item in list_widget.full_list if text.lower() in item.lower()]
		
		# Очищаем текущий виджет
		list_widget.clear()
		list_widget.addItems(filtered_items)
	
	def clear_selection(self, list_widget):
		"""
			Снимает выделение и восстанавливает полный список элементов.
		"""
		# Сбрасываем выбор
		list_widget.clearSelection()
		
		# Восстанавливаем полный список элементов
		if hasattr(list_widget, 'full_list'):
			list_widget.clear()
			list_widget.addItems(list_widget.full_list)
	
	def update_data(self, filtered_df):
		if filtered_df.empty:
			self.clear_list_widgets()
			return
		
		self.filtered_df = filtered_df
		
		for col, (list_widget, search_entry) in self.list_widgets.items():
			list_widget.clear()
			unique_values = sorted(filtered_df[col].dropna().unique())
			list_widget.addItems([str(value) for value in unique_values])
			search_entry.clear()
			
			list_widget.full_list = [str(value) for value in unique_values]
	
	def clear_list_widgets(self):
		for list_widget, _ in self.list_widgets.values():
			list_widget.clear()
	
	def show_filtered_data(self):
		if self.filtered_df.empty:
			QMessageBox.warning(self, "Ошибка", "Нет данных для отображения.")
			return
		
		filtered_df = self.filtered_df.copy()
		for col, (list_widget, _) in self.list_widgets.items():
			selected_items = [item.text() for item in list_widget.selectedItems()]
			if selected_items:
				filtered_df = filtered_df[filtered_df[col].isin(selected_items)]
		
		if filtered_df.empty:
			QMessageBox.warning(self, "Ошибка", "Нет данных, соответствующих выбранным критериям.")
			return
		
		self.filtered_df_to_analyze = filtered_df
		
		dialog = QDialog(self)
		dialog.setWindowTitle("Отфильтрованные данные")
		layout = QVBoxLayout(dialog)
		
		table_view = QTableView()
		model = PandasModel(filtered_df)
		table_view.setModel(model)
		table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
		layout.addWidget(table_view)
		
		analyze_button = QPushButton('Передать данные на анализ')
		analyze_button.clicked.connect(self.send_data_to_analysis)
		layout.addWidget(analyze_button)
		
		expand_button = QPushButton('Развернуть')
		expand_button.setCheckable(True)
		expand_button.toggled.connect(lambda checked: self.toggle_fullscreen(dialog, checked))
		layout.addWidget(expand_button)
		
		dialog.setLayout(layout)
		dialog.resize(800, 600)
		dialog.exec_()
	
	def toggle_fullscreen(self, dialog, checked):
		if checked:
			dialog.showMaximized()
		else:
			dialog.showNormal()
			
	def on_filtered_contracts_received(self, filtered_df):
		""" Слот для обновления данных на основе сигнала от Tab3 """
		print("Получены фильтрованные данные из Tab3")
		print("on_filtered_contracts_received вызван")
		if not isinstance(filtered_df, pd.DataFrame):
			print("Ошибка: Данные не являются DataFrame")
			return
		print(f"Получен DataFrame с {len(filtered_df)} строками и колонками: {filtered_df.columns.tolist()}")
		self.update_data(filtered_df)
