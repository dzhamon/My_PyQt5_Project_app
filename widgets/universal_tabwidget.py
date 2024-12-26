from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal
import pandas as pd

class UniversalTabWidget(QWidget):
    filtered_data_changed = pyqtSignal(pd.DataFrame)  # Сигнал для передачи отфильтрованных данных

    def __init__(self, df, filter_columns):
        super().__init__()
        self.df = df  # DataFrame, который будет либо data_df, либо contract_df
        self.filter_columns = filter_columns  # Список столбцов для фильтрации
        print(f"UniversalTabWidget: Инициализирован с данными и фильтрацией по {filter_columns}")
        # Логика отображения и фильтрации данных

    def apply_filter(self, filter_values):
        """Метод для применения фильтрации данных по переданным значениям."""
        filtered_df = self.df[self.df[self.filter_columns].isin(filter_values)]
        print("UniversalTabWidget: Данные отфильтрованы")
        self.filtered_data_changed.emit(filtered_df)
