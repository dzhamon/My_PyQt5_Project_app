from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget,
                             QLabel, QLineEdit, QListWidget, QScrollArea)
import pandas as pd
import sqlite3
import sys

# блок вызова наших модулей и методов
from utils.config import SQL_PATH
from utils.functions_from041124 import del_nan, calc_indicators, prepare_main_datas
from utils.functions_from041124 import create_treeview_table, scroll_box, param_query
from utils.logic import connect_to_database

pd.options.display.float_format = '{:,.2f}'.format


class SelectFrame:
    print("Проходка 8")
    def __init__(self):
        self.data = {}
        # self.mywindow = mywindow

        # Add a new tab in the QTabWidget
        # self.tab = QWidget()
        # self.mywindow.notebook.addTab(self.tab, "Основные данные")

        # data_df = self.mywindow.data_df
        # self.columns_name = data_df.columns

        # Example queries
        query = "SELECT DISTINCT(lot_number) FROM data_kp;"
        self.number_lots = calc_indicators(query)

        query = "select DISTINCT(actor_name) from data_kp order by actor_name"
        self.actor_names = calc_indicators(query)

        query = "select DISTINCT(discipline) from data_kp order by discipline"
        self.discipline_names = calc_indicators(query)

        query = "select DISTINCT(project_name) from data_kp order by project_name"
        self.project_names = calc_indicators(query)

        query = "select DISTINCT(winner_name) from data_kp order by winner_name"
        self.contragent_winners = calc_indicators(query)

        query = "select DISTINCT(currency) from data_kp where currency not null order by currency"
        self.currency_names = calc_indicators(query)

