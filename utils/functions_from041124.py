import pandas as pd
import numpy as np
import sqlite3
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal, QObject
from utils.config import SQL_PATH


class DataLoader(QObject):
	data_ready = pyqtSignal(pd.DataFrame)

def cleanDataDF(data_df):
	# Преобразуем столбцы в числовой формат, некорректные значения станут NaN
	data_df['total_price'] = pd.to_numeric(data_df['total_price'], errors='coerce')
	data_df['unit_price'] = pd.to_numeric(data_df['unit_price'], errors='coerce')
	
	# Удаляем строки с NaN или нулевыми значениями в 'total_price' и 'unit_price'
	data_df = data_df.dropna(subset=['total_price', 'unit_price'])
	data_df = data_df[data_df['total_price'] > 0]
	data_df = data_df[data_df['unit_price'] > 0]
	
	# Замена разных написаний на единое название для компании
	data_df['winner_name'] = data_df['winner_name'].replace({
		'Не использовать V L GALPERIN NOMIDAGI TOSHKENT TRUBA ZAVODI СП ООО': 'СП "Ташкент трубный завод"',
		'[Удалено]СП "Ташкентский трубный завод"': 'СП "Ташкент трубный завод"',
		'СП "Ташкентский трубный завод"': 'СП "Ташкент трубный завод"',
		'V L GALPERIN NOMIDAGI TOSHKENT TRUBA ZAVODI СП ООО': 'СП "Ташкент трубный завод"'
	})
	# Удаляем лидирующие и замыкающие пробелы из всех текстовых полей
	data_df = data_df.map(lambda x: x.strip() if isinstance(x, str) else x)
	
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
	data_df = pd.read_sql_query(query, conn)
	conn.close()
	return data_df

class DataLoader(QObject):
	data_ready = pyqtSignal(pd.DataFrame, pd.DataFrame, dict)  # Определяем сигнал для передачи DataFrame
	def load_data_contract_from_sql(self):
		print('Загружаются контракты из базы данных')
		conn = sqlite3.connect(SQL_PATH)
		
		# Загрузка данных из базы
		df_c = pd.read_sql_query("SELECT * FROM data_contract ORDER BY contract_signing_date", conn)
		conn.close()
		
		# Заполнение пропусков в числовых полях
		# Преобразуем значения в числовой формат, несовместимые значения станут NaN
		df_c['unit_price'] = pd.to_numeric(df_c['unit_price'], errors='coerce')
		df_c.fillna(df_c['unit_price'].median(), inplace=True)  # Заполняем пропущенные значения медианой
		
		# Удаляем строки с NaN или нулевыми значениями в 'total_price' и 'unit_price'
		df_c = df_c.dropna(subset=['total_contract_amount', 'product_amount', 'unit_price', 'quantity'])
		df_c = df_c[df_c['total_contract_amount'] > 0]
		df_c = df_c[df_c['unit_price'] > 0]
		df_c = df_c[df_c['product_amount'] > 0]
		df_c = df_c[df_c['quantity'] > 0]
		# Преобразуем колонку contract_signing_date и lot_end_date в формат datetime, игнорируя ошибки
		df_c['contract_signing_date'] = pd.to_datetime(df_c['contract_signing_date'], format='%Y-%m-%d', errors='coerce')
		df_c['lot_end_date'] = pd.to_datetime(df_c['lot_end_date'], format='%Y-%m-%d', errors='coerce')
		
		df_c.loc[:, 'executor_dak'] = df_c['executor_dak'].astype(str)
		df_c['executor_dak'] = df_c['executor_dak'].apply(
			lambda x: x.partition(' (')[0] if pd.notna(x) and x != '' else x)
		
		# Замена разных написаний на единое название для компании
		df_c['counterparty_name'] = df_c['counterparty_name'].replace({
			'Не использовать V L GALPERIN NOMIDAGI TOSHKENT TRUBA ZAVODI СП ООО': 'СП "Ташкент трубный завод"',
			'[Удалено]СП "Ташкентский трубный завод"': 'СП "Ташкент трубный завод"',
			'СП "Ташкентский трубный завод"': 'СП "Ташкент трубный завод"',
			'V L GALPERIN NOMIDAGI TOSHKENT TRUBA ZAVODI СП ООО': 'СП "Ташкент трубный завод"'
		})

		# получаем текущую дату
		self.now = pd.Timestamp.now()
		self.contract_df = df_c
		# Количество контрактов в базе данных
		contracts_count = df_c.shape[0]
		# Подсчет строк с датой подписания больше текущей даты
		future_dates_count = df_c[df_c['contract_signing_date'] > self.now].shape[0]
		# Подсчет строк с годом подписания меньше 1900
		invalid_year_count = df_c[df_c['contract_signing_date'].dt.year < 1900].shape[0]
		# Подсчет строк с пустыми значениями unit_price
		missing_unit_price_count = df_c['unit_price'].isnull().sum()
		# Подсчет строк с отрицательной ценой
		negative_price_count = df_c[df_c['unit_price'] < 0].shape[0]
		# Подсчет строк, где дата подписания контракта меньше даты окончания лота
		invalid_signing_date_count = df_c[df_c['contract_signing_date'] < df_c['lot_end_date']].shape[0]
		invalid_dates_df = df_c[df_c['contract_signing_date'] < df_c['lot_end_date']]
		# contracts_less_date(invalid_dates_df)
		invalid_dates_df = df_c[df_c['contract_signing_date'] < df_c['lot_end_date']]
		# Подсчет количества строк с отсутствующими данными executor_dak
		missing_executor_dak_count = df_c['executor_dak'].isnull().sum()
		# Заполнение пропусков в текстовых полях
		df_c.fillna({'executor_dak': 'Не указано'}, inplace=True)
		df_c.fillna({'supplier_unit': 'Не указано'}, inplace=True)
		
		# Сбор статистики
		stats = {
			'contracts_count': df_c.shape[0],
			'future_dates_count': df_c[df_c['contract_signing_date'] > pd.Timestamp.now()].shape[0],
			'invalid_year_count': df_c[df_c['contract_signing_date'].dt.year < 1900].shape[0],
			'missing_unit_price_count': df_c['unit_price'].isnull().sum(),
			'negative_price_count': df_c[df_c['unit_price'] < 0].shape[0],
			'invalid_signing_date_count': df_c[df_c['contract_signing_date'] < df_c['lot_end_date']].shape[0],
			'missing_executor_dak_count': df_c['executor_dak'].isnull().sum()
		}
		
		# Испускаем сигнал с DataFrame
		self.data_ready.emit(df_c, invalid_dates_df, stats)
		
		
	# Замена некорректных дат подписания контрактов (если дата больше текущей или год < 1900)
def fix_signing_date(row):
	if pd.isnull(row['contract_signing_date']) or not isinstance(row['contract_signing_date'], pd.Timestamp):
		return row['lot_end_date']  # заменяем на дату завершения лота
	
	# Проверяем, что дата подписания контракта корректна (не в будущем и не слишком старая)
	if row['contract_signing_date'] > self.now or row['contract_signing_date'].year < 1900:
		return row['lot_end_date']  # Заменяем на дату завершения лота
	
	# Проверка, чтобы дата подписания контракта не была раньше даты завершения лота
	if row['contract_signing_date'] < row['lot_end_date']:
		return row['lot_end_date'] + pd.Timedelta(
			days=10)  # Заменяем на дату завершения лота + 10 дней, если контракт подписан раньше
	
	return row['contract_signing_date']

	# Применение функции к каждой строке для исправления дат
	df_c['contract_signing_date'] = df_c.apply(fix_signing_date, axis=1)
	
	return (df_c, contracts_count, future_dates_count, invalid_year_count, missing_unit_price_count,
	        negative_price_count, invalid_signing_date_count, missing_executor_dak_count)


def del_nan(list_name):
	L1 = [item for item in list_name if not (pd.isnull(item)) is True]
	L1, list_name = list_name, L1
	return list_name


def get_unique_only(st): # Этот метод более не исползуется
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


# Перевод данных в единую валюту USD
def convert_to_eur(data_df):
	"""
	Функция для конвертации валют в EUR.
	"""
	exchange_rate_to_eur = {
		'AED': 0.23,
		'CNY': 0.13,
		'EUR': 1.0,  # Базовая валюта
		'GBP': 1.13,
		'KRW': 0.00077,
		'KZT': 0.002,
		'RUB': 0.011,
		'USD': 0.83,
		'UZS': 0.000071,
		'JPY': 0.0073,
		'SGD': 0.61
	}
	data_df = data_df.copy()
	# Преобразуем стоимость в EUR
	data_df['exchange_rate'] = data_df['contract_currency'].map(exchange_rate_to_eur)
	data_df = data_df.dropna(subset=['exchange_rate'])
	data_df['total_contract_amount_eur'] = data_df['total_contract_amount'] * data_df['exchange_rate']
	
	return data_df


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
