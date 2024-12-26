from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QWidget, QListWidget, QMessageBox, QLineEdit,
                             QVBoxLayout, QPushButton, QTableView, QDialog, QAbstractItemView)
from PyQt5.QtCore import pyqtSignal, QTimer
from utils.PandasModel_previous import PandasModel
import pandas as pd


class Tab4(QWidget):
	data_ready_for_analysis = pyqtSignal(pd.DataFrame)
	
	def __init__(self):
		super().__init__()
		# создаем основной макет
		self.layout = QVBoxLayout(self)
		# Словарь для хранения QListWidget и их значений
		self.list_widgets = {}
		self.filtered_df = pd.DataFrame()  # Пустой DataFrame для хранения отфильтрованных данных
		
		# Создание макета для списков (горизонтальный макет)
		list_layout = QHBoxLayout()
		# Инициализация 6 списков и кнопок "Очистить"
		self.init_list_widgets(list_layout)
		
		# Создание кнопки для выполнения запроса
		self.execute_button = QPushButton('Выполнить запрос')
		self.execute_button.clicked.connect(self.show_filtered_data)
		# Добавление макета списков в основной макет
		self.layout.addLayout(list_layout)
		self.layout.addWidget(self.execute_button)
		self.setLayout(self.layout)  # Устанавливаем основной макет
	
	# Подключаем сигнал от Tab3 к методу обновления данных
	# self.filtered_contracts_changed.connect(self.on_filtered_contracts_received)
	
	def on_filtered_contracts_received(self, filtered_df):
		""" Слот для обновления данных на основе сигнала от Tab3 """
		print("Получены фильтрованные данные из Tab3")
		print("on_filtered_contracts_received вызван")
		if not isinstance(filtered_df, pd.DataFrame):
			print("Ошибка: Данные не являются DataFrame")
			return
		print(f"Получен DataFrame с {len(filtered_df)} строками и колонками: {filtered_df.columns.tolist()}")
		self.update_data(filtered_df)
	
	def filter_listbox(self, text, list_widget):
		"""
		    Фильтрует элементы в списке с минимизацией обновлений.
		    """
		print(f"Фильрация списка {list_widget.objectName()} по текстуЖ '{text}'")
		# Сохраняем текущую позицию прокрутки и выбранные элементы
		scroll_position = list_widget.verticalScrollBar().value()
		selected_items = [item.text() for item in list_widget.selectedItems()]
		
		# Собираем список видимых элементов
		visible_items = []
		for row in range(list_widget.count()):
			item = list_widget.item(row)
			if text.lower() in item.text().lower():
				visible_items.append(item.text())
		
		# Очищаем и наполняем виджет новыми элементами
		list_widget.clear()
		list_widget.addItems(visible_items)
		
		# Восстанавливаем выделение
		for row in range(list_widget.count()):
			item = list_widget.item(row)
			if item.text() in selected_items:
				item.setSelected(True)
		
		# Восстанавливаем прокрутку
		list_widget.verticalScrollBar().setValue(scroll_position)


	def update_data(self, filtered_df):
		"""
		Обновляет данные в QListWidget и включает их для выбора.
		"""
		print("update_data вызван")
		print(f"Колонки DataFrame: {filtered_df.columns.tolist()}")
		if filtered_df.empty:
			print("Получен пустой DataFrame.")
			self.clear_list_widgets()
			return
		self.filtered_df = filtered_df
		
		# Заполняем QListWidget уникальными значениями из DataFrame
		# filtered_df.loc[:,'executor_dak'] = filtered_df['executor_dak'].astype(str)
		for col in self.list_widgets:
			if col not in filtered_df.columns:
				print(f"Столбец {col} отсутствует в DataFrame.")
				continue
			list_widget, _ = self.list_widgets[col]
			unique_values = sorted(filtered_df[col].dropna().unique())
			print(f"Для столбца {col} найдены значения: {unique_values[:10]} (первые 10)")
			list_widget.clear()
			list_widget.addItems([str(value) for value in unique_values])
	
	
	def send_data_to_analysis(self):
		"""
			Передает отфильтрованные данные в модуль анализа
			"""
		if hasattr(self, 'filtered_df_to_analyze') and not self.filtered_df_to_analyze.empty:
			self.data_ready_for_analysis.emit(self.filtered_df_to_analyze)
		else:
			QMessageBox.warning(self, "Ошибка", "Нет данных для анализа")
	
	
	def init_list_widgets(self, layout):
		"""
		Инициализирует 6 QListWidget и кнопки "Очистить", добавляет их в горизонтальный макет.
		"""
		list_names = ['lot_number', 'discipline', 'contract_name', 'executor_dak',
		              'counterparty_name', 'product_name', 'contract_currency']
		
		for name in list_names:
			print(f"Инициализация списка для: {name}")
			widget_layout = QVBoxLayout()  # Используем вертикальный макет для метки, списка и кнопки "Очистить"
			label = QLabel(f"Выберите {name}:")
			search_entry = QLineEdit(self)
			search_entry.setPlaceholderText(f"Поиск по {name}...")
			search_entry.show()
			
			list_widget = QListWidget()
			list_widget.setObjectName(name)
			list_widget.setSelectionMode(QListWidget.MultiSelection)  # Поддержка множественного выбора
			list_widget.show()
			
			# Связь строки поиска с методом фильтрации
			search_entry.textChanged.connect(lambda text, lw=list_widget: self.filter_listbox(text, lw))
			
			# Кнопка "Очистить"
			clear_button = QPushButton('Очистить')
			clear_button.clicked.connect(lambda _, lw=list_widget: self.clear_selection(lw))
			
			# Добавление виджетов в макет
			widget_layout.addWidget(label)
			widget_layout.addWidget(search_entry)
			widget_layout.addWidget(list_widget)
			widget_layout.addWidget(clear_button)
			layout.addLayout(widget_layout)
			
			# Сохранение виджетов в словаре
			self.list_widgets[name] = list_widget, search_entry
		
		# Создаем таймер для отложенной фильтрации
		self.filter_timer = QTimer(self)
		self.filter_timer.setSingleShot(True)
		self.filter_timer.timeout.connect(lambda lw=list_widget, se=search_entry:
		                                  self.filter_listbox(se.text(), lw))
		
		# Связываем ввод текста с запуском таймера
		search_entry.textChanged.connect(lambda: self.filter_timer.start(300))
	
	
	def clear_list_widgets(self):
		"""
		Очищает все QListWidget.
		"""
		for list_widget in self.list_widgets.values():
			list_widget.clear()
	
	
	def clear_selection(self, list_widget):
		"""
		Снимает выделение со всех элементов в указанном QListWidget.
		"""
		list_widget.clearSelection()
	
	
	def show_filtered_data(self):
		"""
		Отображает отфильтрованные данные во всплывающем окне.
		"""
		if self.filtered_df.empty:
			QMessageBox.warning(self, "Ошибка", "Нет данных для отображения.")
			return
		
		# Применяем фильтры на основе выбранных элементов в QListWidget
		filtered_df = self.filtered_df.copy()
		for col, (list_widget, _) in self.list_widgets.items():
			selected_items = [item.text() for item in list_widget.selectedItems()]
			if selected_items:
				filtered_df = filtered_df[filtered_df[col].isin(selected_items)]
		
		if filtered_df.empty:
			QMessageBox.warning(self, "Ошибка", "Нет данных, соответствующих выбранным критериям.")
			return
		
		# Сохраняем отфильтрованные данные в атрибут класса для передачи в analysis.py
		self.filtered_df_to_analyze = filtered_df
		
		# Создание всплывающего окна для отображения отфильтрованных данных
		dialog = QDialog(self)
		dialog.setWindowTitle("Отфильтрованные данные")
		layout = QVBoxLayout(dialog)
		
		# Создаем виджет таблицы для отображения данных
		table_view = QTableView()
		model = PandasModel(filtered_df)
		table_view.setModel(model)
		table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
		layout.addWidget(table_view)
		
		# Создание кнопки "Analyze Data"
		analyze_button = QPushButton('Передать данные на анализ')
		analyze_button.clicked.connect(self.send_data_to_analysis)
		layout.addWidget(analyze_button)
		
		# Создание кнопки "Развернуть"
		expand_button = QPushButton('Развернуть')
		expand_button.setCheckable(True)  # Кнопка будет переключаемой
		expand_button.toggled.connect(lambda checked: self.toggle_fullscreen(dialog, checked))
		layout.addWidget(expand_button)
		
		dialog.setLayout(layout)
		dialog.resize(800, 600)
		dialog.exec_()
	
	
	def toggle_fullscreen(self, dialog, checked):
		"""
		Разворачивает или возвращает окно к предыдущему размеру в зависимости от состояния кнопки.
		"""
		if checked:
			dialog.showMaximized()
		else:
			dialog.showNormal()
