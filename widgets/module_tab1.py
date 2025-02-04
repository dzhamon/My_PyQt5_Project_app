from PyQt5.QtCore import pyqtSignal, QDate
from PyQt5.QtWidgets import (QWidget, QGridLayout, QDateEdit, QLabel, QPushButton,
                             QTableView, QFrame, QMessageBox)
import pandas as pd
from dateutil.relativedelta import relativedelta
import sqlite3
from utils.config import SQL_PATH
from utils.PandasModel_previous import PandasModel

class Tab1Widget(QWidget):
    # Определяем сигнал, который будет испускаться при изменении фильтрованных данных
    filtered_data_changed = pyqtSignal(pd.DataFrame)

    def __init__(self, data_df):
        super().__init__()
        self.data_df = data_df  # Сохраняем исходный DataFrame
        self.filtered_df = data_df  # Изначально filtered_df — это весь DataFrame
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

        # Метки и выбор дат
        label = QLabel("Выберите диапазон дат загрузки Лотов")
        self.start_date_edit = QDateEdit(self)
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.currentDate().addMonths(-1))

        self.end_date_edit = QDateEdit(self)
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate())

        # Кнопка применения фильтра
        button = QPushButton("Применить диапазон дат")
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
        button.clicked.connect(self.apply_date_filter)

        frame1_layout.addWidget(label, 0, 0, 1, 2)
        frame1_layout.addWidget(QLabel("Начальная дата:"), 1, 0)
        frame1_layout.addWidget(self.start_date_edit, 1, 1)
        frame1_layout.addWidget(QLabel("Конечная дата:"), 2, 0)
        frame1_layout.addWidget(self.end_date_edit, 2, 1)
        frame1_layout.addWidget(button, 3, 0, 1, 2)

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

    def apply_date_filter(self):
        # Получаем выбранные даты
        start_date = self.start_date_edit.date().toPyDate()
        end_date = self.end_date_edit.date().toPyDate()
        
        # Преобразуем даты в datetime64[ns]
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        if start_date > end_date:
            QMessageBox.warning(self, "Предупреждение",
                                "Начальная дата позже конечной даты. Проверьте корректность значений")
            return

        self.filtered_df = self.filter_data_by_range(self.data_df, start_date, end_date)
        self.display_data(self.filtered_df)
        self.filtered_data_changed.emit(self.filtered_df)  # Испускаем сигнал с новыми данными

    def filter_data_by_range(self, data_df, start_date, end_date):
        # Фильтрация данных по диапазону дат
        data_df['close_date'] = pd.to_datetime(data_df['close_date'], errors='coerce')
        data_df = data_df.dropna(subset=['close_date'])

        filtered_df = data_df[(data_df['close_date'] >= start_date) & (data_df['close_date'] <= end_date)]
        return filtered_df

    def display_data(self, df):
        if df.empty:
            QMessageBox.warning(self, "Ошибка", "DataFrame пустой, нечего отображать.")
            return
        model = PandasModel(df)
        self.table_widget.setModel(model)
        self.table_widget.setSortingEnabled(True)
        # DataFrame отображен в QTableView
