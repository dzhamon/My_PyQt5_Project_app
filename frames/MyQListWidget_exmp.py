from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QScrollArea, QTabWidget, QLabel, QFrame)
from PyQt5.QtCore import Qt
import pandas as pd


class MyTabWidget(QWidget):
	def __init__(self):
		super().__init__()
		self.layout = QVBoxLayout(self)
		
		# Создаем QTabWidget
		self.tabs = QTabWidget()
		self.tab1 = QWidget()
		self.tabs.addTab(self.tab1, "Метрики")
		self.layout.addWidget(self.tabs)
		self.setLayout(self.layout)
		
		# Инициализируем вкладку
		self.create_tab1(self.tab1)
	
	def create_tab1(self, tab):
		# Основной макет для вкладки
		tab_layout = QVBoxLayout(tab)
		
		# Создаем QScrollArea
		scroll_area = QScrollArea()
		scroll_area.setWidgetResizable(True)  # Позволяет QScrollArea изменять размер виджета
		
		# Создаем виджет для содержимого в QScrollArea
		scroll_widget = QWidget()
		scroll_layout = QVBoxLayout(scroll_widget)
		
		# Создаем несколько QListWidget
		for i in range(3):  # Пример создания 3 списков
			list_widget = QListWidget()
			# Наполняем QListWidget данными
			list_widget.addItems(
				self.get_sample_data())  # Замените get_sample_data() на реальный метод получения данных
			scroll_layout.addWidget(list_widget)
		
		# Добавляем виджет содержимого в QScrollArea
		scroll_area.setWidget(scroll_widget)
		tab_layout.addWidget(scroll_area)
	
	def get_sample_data(self):
		# Здесь вы можете возвращать данные из вашего DataFrame
		return ["Item 1", "Item 2", "Item 3"]  # Пример данных


# Вызов класса MyTabWidget
if __name__ == "__main__":
	from PyQt5.QtWidgets import QApplication
	import sys
	
	app = QApplication(sys.argv)
	window = MyTabWidget()
	window.show()
	sys.exit(app.exec_())
