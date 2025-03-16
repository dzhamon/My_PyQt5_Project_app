import pandas as pd
import numpy as np
import sqlite3
from PyQt5.QtWidgets import QMessageBox
import json
from utils.config import SQL_PATH
import matplotlib.pyplot as plt
import seaborn as sns

import os
import re

_cached_data = None


def cleanDataDF(data_df):
	# Удаляем пробелы (разделители тысяч) и заменяем запятую на точку (десятичный разделитель)
	data_df['total_price'] = pd.to_numeric(
		data_df['total_price'].astype(str).str.replace(' ', '', regex=False).str.replace(',', '.', regex=False),
		errors='coerce')
	data_df['unit_price'] = pd.to_numeric(
		data_df['unit_price'].astype(str).str.replace(' ', '', regex=False).str.replace(',', '.', regex=False),
		errors='coerce')
	
	# Преобразуем столбцы в числовой формат
	data_df['total_price'] = pd.to_numeric(data_df['total_price'], errors='coerce')
	data_df['unit_price'] = pd.to_numeric(data_df['unit_price'], errors='coerce')
	
	# Удаляем строки с NaN в ключевых столбцах
	data_df = data_df.dropna(subset=['total_price', 'unit_price'])
	data_df = data_df[data_df['total_price'] > 0]
	data_df = data_df[data_df['unit_price'] > 0]
	
	# Нормализация названий компаний
	data_df.loc[:, 'winner_name'] = data_df['winner_name'].replace({
		'Не использовать V L GALPERIN NOMIDAGI TOSHKENT TRUBA ZAVODI СП ООО': 'СП "Ташкент трубный завод"',
		'[Удалено]СП "Ташкентский трубный завод"': 'СП "Ташкент трубный завод"',
		'СП "Ташкентский трубный завод"': 'СП "Ташкент трубный завод"',
		'V L GALPERIN NOMIDAGI TOSHKENT TRUBA ZAVODI СП ООО': 'СП "Ташкент трубный завод"',
		'ТСК-РЕГИОН ООО': 'ООО ТСК-РЕГИОН',
		'ТСК РЕГИОН': 'ООО ТСК-РЕГИОН',
		'ТСК Регион ': 'ООО ТСК-РЕГИОН',
		'ООО "УПСК-экспорт"': 'УПСК-ЭКСПОРТ',
		'УПСК-ЭКСПОРТ ООО': 'УПСК-ЭКСПОРТ',
		'не использовать УПСК-экспорт ООО': 'УПСК-ЭКСПОРТ',
		'ООО "Темир Бетон Конструкциялари Комбинати"': 'ООО "Темир Бетон',
		'не использовать DREAM-ALLIANCE': 'DREAM ALLIANCE',
		'не использовать NEW FORMAT TASHKENT': 'NEW FORMAT TASHKENT',
		'не использовать NASIBA GAVHAR': 'NASIRA GAVHAR',
		'Daromad Munira Fayz Textile<не использовать>': 'Daromad Munira Fayz'
	})
	
	# Применяем `apply` к строковым столбцам, удаляя пробелы в начале и конце строк
	data_df = data_df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
	
	return data_df


def load_data_from_sql():
	print("Загружаем все Лоты из базы данных")
	conn = sqlite3.connect(SQL_PATH)
	cur = conn.cursor()
	cur.execute(
		"DELETE FROM data_kp WHERE (discipline IS NULL OR TRIM(discipline) = '') OR (currency IS NULL OR TRIM(currency) = '')")
	conn.commit()
	query = """
	       SELECT lot_number, discipline, project_name,
	              open_date, close_date, actor_name, good_name,
	              good_count, unit, supplier_qty, supplier_unit,
	              winner_name, unit_price, total_price, currency
	       FROM data_kp
	       ORDER BY close_date;
	   """
	df = pd.read_sql_query(query, conn)
	conn.close()
	return df


def clean_contract_data(df_c):
	# метод очистки данных по контрактам
	
	# Удаляем пробелы (разделители тысяч) и заменяем запятую на точку (десятичный разделитель)
	df_c['total_contract_amount'] = pd.to_numeric(
		df_c['total_contract_amount'].astype(str).str.replace(' ', '', regex=False).str.replace(',', '.', regex=False),
		errors='coerce')
	df_c['unit_price'] = pd.to_numeric(
		df_c['unit_price'].astype(str).str.replace(' ', '', regex=False).str.replace(',', '.', regex=False),
		errors='coerce')
	# Преобразуем значения в числовой формат, несовместимые значения станут NaN
	df_c['unit_price'] = pd.to_numeric(df_c['unit_price'], errors='coerce')
	df_c.fillna(df_c['unit_price'].median(), inplace=True)  # Заполняем пропущенные значения медианой
	
	# Удаляем строки с NaN или нулевыми значениями в 'total_price' и 'unit_price'
	df_c = df_c.dropna(subset=['total_contract_amount', 'product_amount', 'unit_price', 'quantity'])
	df_c = df_c[df_c['total_contract_amount'] > 0]
	df_c = df_c[df_c['unit_price'] > 0]
	df_c = df_c[df_c['product_amount'] > 0]
	df_c = df_c[df_c['quantity'] > 0]
	# Преобразуем колонку contract_signing_date в формат datetime, игнорируя ошибки
	# Преобразуем колонку contract_signing_date и lot_end_date в формат datetime, игнорируя ошибки
	df_c['contract_signing_date'] = pd.to_datetime(df_c['contract_signing_date'], format='%Y-%m-%d', errors='coerce')
	df_c['lot_end_date'] = pd.to_datetime(df_c['lot_end_date'], format='%Y-%m-%d', errors='coerce')
	
	# Замена разных написаний на единое название для компании
	df_c['counterparty_name'] = df_c['counterparty_name'].replace({
		'Не использовать V L GALPERIN NOMIDAGI TOSHKENT TRUBA ZAVODI СП ООО': 'СП "Ташкент трубный завод"',
		'[Удалено]СП "Ташкентский трубный завод"': 'СП "Ташкент трубный завод"',
		'СП "Ташкентский трубный завод"': 'СП "Ташкент трубный завод"',
		'V L GALPERIN NOMIDAGI TOSHKENT TRUBA ZAVODI СП ООО': 'СП "Ташкент трубный завод"'
	})
	
	# получаем текущую дату
	now = pd.Timestamp.now()
	# Количество контрактов в базе данных
	contracts_count = df_c.shape[0]
	
	# Подсчет строк с датой подписания больше текущей даты
	future_dates_count = df_c[df_c['contract_signing_date'] > now].shape[0]
	
	# Подсчет строк с годом подписания меньше 1900
	invalid_year_count = df_c[df_c['contract_signing_date'].dt.year < 1900].shape[0]
	
	# Подсчет строк с пустыми значениями unit_price
	missing_unit_price_count = df_c['unit_price'].isnull().sum()
	
	# Подсчет строк с отрицательной ценой
	negative_price_count = df_c[df_c['unit_price'] < 0].shape[0]
	
	# Подсчет строк, где дата подписания контракта меньше даты окончания лота
	invalid_signing_date_count = df_c[df_c['contract_signing_date'] < df_c['lot_end_date']].shape[0]
	invalid_dates_df = df_c[df_c['contract_signing_date'] < df_c['lot_end_date']]
	
	# Подсчет количества строк с отсутствующими данными executor_dak
	missing_executor_dak_count = df_c['executor_dak'].isnull().sum()
	
	# Заполнение пропусков в текстовых полях
	df_c.fillna({'executor_dak': 'Не указано'}, inplace=True)
	df_c.fillna({'supplier_unit': 'Не указано'}, inplace=True)
	
	df_c['lot_end_date'] = pd.to_datetime(df_c['lot_end_date'], errors='coerce')
	df_c['contract_signing_date'] = pd.to_datetime(df_c['contract_signing_date'], errors='coerce')
	
	# Замена некорректных дат подписания контрактов (если дата больше текущей или год < 1900)
	
	def fix_signing_date(row):
		if pd.isnull(row['contract_signing_date']) or not isinstance(row['contract_signing_date'], pd.Timestamp):
			return row['lot_end_date']  # заменяем на дату завершения лота
		
		# Проверяем, что дата подписания контракта корректна (не в будущем и не слишком старая)
		if row['contract_signing_date'] > now or row['contract_signing_date'].year < 1900:
			return row['lot_end_date']  # Заменяем на дату завершения лота
		
		# Проверка, чтобы дата подписания контракта не была раньше даты завершения лота
		if row['contract_signing_date'] < row['lot_end_date']:
			return row['lot_end_date'] + pd.Timedelta(
				days=10)  # Заменяем на дату завершения лота + 10 дней, если контракт подписан раньше
		
		return row['contract_signing_date']
	
	# Применение функции к каждой строке для исправления дат
	df_c['contract_signing_date'] = df_c.apply(fix_signing_date, axis=1)
	
	# Сохранить данные в кэш
	global _cached_data
	_cached_data = {
		"invalid_dates_df": invalid_dates_df,
		"contract_df": df_c
	}

	return df_c

def get_cached_data():
    global _cached_data
    if _cached_data is not None:
        print("Данные получены из кэша")
        return _cached_data["invalid_dates_df"], _cached_data["contract_df"]
    else:
        return load_data_contract_from_sql()


def del_nan(list_name):
	L1 = [item for item in list_name if not (pd.isnull(item)) is True]
	L1, list_name = list_name, L1
	return list_name


def get_unique_only(st):
	# Empty list
	lst1 = []
	count = 0
	# traverse the array
	for i in st:
		if i != 0:
			if i not in lst1:
				count += 1
				lst1.append(i)
	return lst1


# Функция "обрезки" строки до нужного символа
def cut_list(lstt_act):
	last_act = []
	for lst_act in lstt_act:
		try:
			if pd.notna(lst_act) and lst_act != '':
				last_act.append(lst_act.partition(' (')[0])
			else:
				last_act.append(np.nan)  # добавление NaN для соответствия длине
		except AttributeError:
			last_act.append(np.nan)  # добавление NaN при ошибке
	return last_act


def calc_indicators(query):
	conn = sqlite3.connect(SQL_PATH)
	cur = conn.cursor()
	res = cur.execute(query).fetchall()
	return res


def prepare_main_datas(sql_query=None):
	# Суммы и средние значения контрактов в разрезе Дисциплин и валют контрактов
	# материал по работе SQLite_Python заимствован из
	# https://sky.pro/wiki/sql/preobrazovanie-rezultatov-zaprosa-sqlite-v-slovar/
	conn = sqlite3.connect(SQL_PATH)
	cur = conn.cursor()
	cur.execute(sql_query)
	columns = [column[0] for column in cur.description]
	values = cur.fetchall()
	row_dict = {}
	k = 0
	for column in columns:
		list_tmp = []
		for value in values:
			list_tmp.append(value[k])
		row_dict[column] = list_tmp
		k += 1
	df = pd.DataFrame(row_dict)
	return df


def create_treeview_table(df):
	columns = df.columns
	print('Our DF columns is ', columns)
	list_of_rows = []
	print('df.shape =', df.shape)
	for i in range(df.shape[0]):
		list_of_rows.append(df.T[i].tolist())
	param1 = columns
	param2 = list_of_rows
	return param1, param2


# функция параметризации запроса
def param_query(qry):
	conn = sqlite3.connect(SQL_PATH)
	cur = conn.cursor()
	cur.execute(qry)
	
	print(cur.fetchall())
	
	conn.close()


def trim_actor_name(name):
	"""
	Обрезает строку до первой открывающей скобки.
	Например, 'Алишеров Асадбек Абдулла угли (вн. 31902) (моб. +998936457575)' станет 'Алишеров Асадбек Абдулла угли'.
	"""
	return name.split('(')[0].strip()


class CurrencyConverter:
	"""
	Класс для конвертации валют в евро.
	"""
	
	def __init__(self, exchange_rates=None):
		"""
		Инициализация курсов валют.
		:param exchange_rates: Словарь с курсами валют к EUR.
		"""
		# Курсы валют по умолчанию
		self.exchange_rates = exchange_rates or {
			'AED': 0.23, 'CNY': 0.13, 'EUR': 1.0, 'GBP': 1.13,
			'KRW': 0.00077, 'KZT': 0.002, 'RUB': 0.011, 'USD': 0.83,
			'UZS': 0.000071, 'JPY': 0.0073, 'SGD': 0.61
		}
	
	def update_rates(self, new_rates):
		"""
		Обновляет курсы валют.
		:param new_rates: Словарь с новыми курсами валют.
		"""
		self.exchange_rates.update(new_rates)
	
	def convert_column(self, df, amount_column, currency_column, result_column=None):
		"""
		Конвертирует значения одного столбца суммы в EUR.
		:param df: DataFrame с данными.
		:param amount_column: Столбец с суммой.
		:param currency_column: Столбец с валютами.
		:param result_column: Столбец для сохранения результата (если None, перезаписывает amount_column).
		:return: Обновленный DataFrame.
		"""
		result_column = result_column or amount_column
		df = df.copy()
		
		# Добавляем столбец с курсами валют
		df['exchange_rate'] = df[currency_column].map(self.exchange_rates)
		df = df.dropna(subset=['exchange_rate'])  # Удаляем строки с неизвестными валютами
		
		# Конвертируем сумму
		df[result_column] = df[amount_column] * df['exchange_rate']
		return df.drop(columns=['exchange_rate'])
	
	def convert_multiple_columns(self, df, columns_info):
		"""
		Конвертирует несколько столбцов в EUR.
		:param df: DataFrame с данными.
		:param columns_info: Список кортежей [(amount_column, currency_column, result_column), ...].
		:return: Обновленный DataFrame.
		"""
		df = df.copy()
		for amount_column, currency_column, result_column in columns_info:
			df = self.convert_column(df, amount_column, currency_column, result_column)
		return df


def save_analysis_results(analysis_results, output_path):
	try:
		analysis_results.to_excel(output_path, index=False)
		print("Файл успешно сохранен.")
	
	except PermissionError:
		msg = QMessageBox()
		msg.setIcon(QMessageBox.Warning)
		msg.setWindowTitle("Ошибка")
		msg.setText("Файл открыт. Закройте файл и попробуйте снова.")
		msg.exec_()
	return

# Функция для загрузки подсказок для JSON-файла
def load_menu_hints():
	with open('menu_hints.json', 'r', encoding='utf-8') as file:
		return json.load(file)

menu_hints = load_menu_hints()

# очистка имени файла от запрещенных символов
def clean_filename(filename):
    """
    Очищает имя файла от недопустимых символов.
    Заменяет запрещённые символы на "_" и убирает лишние пробелы.
    """
    # Заменяем недопустимые символы на "_"
    cleaned = re.sub(r'[\\/*?:"<>|()]', '_', filename)
    # Убираем лишние пробелы
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def plot_supplier_prices_by_currency(results_by_currency):
	"""
	   Построение графиков для каждого валютного анализа.
	   Args:
	       results_by_currency (dict): Результаты анализа, сгруппированные по валютам.
	"""
	print("Началось рисование графиков")
	for currency, data in results_by_currency.items():
		if data['top_suppliers'].empty or data['bottom_suppliers'].empty:
			print(f"Нет данных для валюты {currency}. Пропускаем ...")
			continue
			
		top_suppliers = data['top_suppliers']
		bottom_suppliers = data['bottom_suppliers']
		
		# График для топ-10 поставщиков с высокими ценами
		plt.figure(figsize=(10, 6))
		sns.barplot(
			data=top_suppliers,
			x='avg_unit_price',
			y='winner_name',
			hue='winner_name',
			dodge=False,
			palette='Reds_r',
			legend=False
		)
		
		plt.title(f"Топ-10 поставщиков с высокими ценами ({currency})")
		plt.xlabel("Средняя цена за единицу")
		plt.ylabel("Поставщик")
		plt.tight_layout()
		plt.show()
		
		# График для топ-10 поставщиков с низкими ценами
		plt.figure(figsize=(10, 6))
		sns.barplot(
			data=bottom_suppliers,
			x='avg_unit_price',
			y='winner_name',
			hue='winner_name',
			dodge=False,
			palette='Greens',
			legend=False
		)
		plt.title(f"Топ-10 поставщиков с низкими ценами ({currency})")
		plt.xlabel("Средняя цена за единицу")
		plt.ylabel("Поставщик")
		plt.tight_layout()
		plt.show()
	
def check_file_access(file_path):
    """
    Проверяет доступ к файлу. Если файл открыт, выводит сообщение об ошибке.
    """
    try:
        # Пробуем открыть файл для записи
        with open(file_path, 'a'):
            pass
        return True
    except IOError:
        # Если файл занят, выводим сообщение
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Ошибка доступа к файлу")
        msg_box.setText(f"Файл {file_path} открыт в другом приложении.\nПожалуйста, закройте его.")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
        return False
