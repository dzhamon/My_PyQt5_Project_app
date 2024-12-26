# analysis_widget.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton


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
	
	# Ваш код для кластерного анализа здесь...
	
	def perform_monthly_expense_analysis(self):
		print("Выполняем анализ месячных затрат.")
	
	# Ваш код для анализа месячных затрат здесь...
	
	def perform_other_analysis(self):
		print("Выполняем другой анализ.")
	# Ваш код для другого анализа здесь...
