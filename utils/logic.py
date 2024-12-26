import pandas as pd
import numpy as np
from collections import Counter
from utils.config import SQL_PATH
from utils.functions import del_nan
from datetime import datetime
import sqlite3


# Этот метод не используется. Есть более круче от самого Python basic_df = basic_df.drop_duplicates(subset='lot_number')
def del_double_rows(basic_df):
	number_lots = del_nan(set(basic_df['lot_number']))
	for number_lot in number_lots:
		df_vrem = basic_df.loc[basic_df['lot_number'] == number_lot]
		if len(df_vrem) > 1:
			list_tup = []
			for ind in range(len(df_vrem)):
				ssl = df_vrem.iloc[[ind][0]].to_list()
				ssl = tuple(ssl)
				list_tup.append(ssl)
			c = Counter(list_tup)
			if set(c.values()) == {1}:
				continue
			else:
				list_unique_tup = set(list_tup)
				df_vrem = pd.DataFrame(list_unique_tup, columns=basic_df.columns)
				# удаляем из excel_data_df строки по номеру лота (number_lot)
				basic_df = basic_df[basic_df['lot_number'] != number_lot]
				basic_df = pd.concat([basic_df, df_vrem], ignore_index=True)
	return basic_df


def clean_data_from_xls(file):
	# start_time = time.time()
	
	dict_names = {'Номер лота': 'lot_number',
	              'Статус лота': 'lot_status',
	              'Дисциплина': 'discipline',
	              'Наименование проекта': 'project_name',
	              'Дата открытия лота': 'open_date',
	              'Дата закрытия лота': 'close_date',
	              'Исполнитель МТО (Ф.И.О.)': 'actor_name',
	              'Наименование ТМЦ': 'good_name',
	              'Количество ТМЦ': 'good_count',
	              'Ед. изм. ТМЦ': 'unit',
	              'Кол-во поставщика': 'supplier_qty',
	              'Ед.изм. поставщика': 'supplier_unit',
	              'Присуждено контрагенту': 'winner_name',
	              'Цена': 'unit_price',
	              'Сумма контракта': 'total_price',
	              'Валюты контракта': 'currency'}
	
	excel_data_df = pd.read_excel(file)
	excel_data_df = excel_data_df.rename(columns=dict_names)
	
	# удаляем дублирующиеся строки в датафрейме
	excel_data_df = excel_data_df.drop_duplicates(subset='lot_number')
	
	# заменим в числовых полях excel_data_df все отсутствующие данные (nan) на ноль (0)
	excel_data_df['good_count'] = excel_data_df['good_count'].replace(np.nan, 0)
	excel_data_df['total_price'] = excel_data_df['total_price'].replace(np.nan, 0)
	excel_data_df['supplier_qty'] = excel_data_df['supplier_qty'].replace(np.nan, 0)
	excel_data_df['unit_price'] = excel_data_df['unit_price'].replace(np.nan, 0)
	
	# # Перед присвоением проверяем размеры
	# print("Размер DataFrame excel_data_df:", len(excel_data_df))
	# processed_actor_name = cut_list(excel_data_df['actor_name'])
	# print("Размер обработанного списка processed_actor_name:", len(processed_actor_name))
	#
	# # Убедитесь, что длины совпадают перед присвоением
	# if len(processed_actor_name) == len(excel_data_df):
	# 	excel_data_df['actor_name'] = processed_actor_name
	# else:
	# 	print("Ошибка: размеры не совпадают!")
	
	# excel_data_df['actor_name'] = cut_list(excel_data_df['actor_name'])
	excel_data_df['actor_name'] = excel_data_df['actor_name'].apply(
		lambda x: x.partition(' (')[0] if pd.notna(x) and x != '' else x)
	
	return excel_data_df


def clean_contr_data_from_xls(file_path):
	# Создадим словарь наимеований столбцов
	dict_contract = {'Номер лота': 'lot_number',
	                 'Дата завершения лота': 'lot_end_date',
	                 'Номер контракта/договора по этому лоту': 'contract_number',
	                 'Дата заключения контракта/договора': 'contract_signing_date',
	                 'Наименование контракта/договора': 'contract_name',
	                 'Исполнитель ДАК': 'executor_dak',
	                 'Наименование контрагента-владельца контракта/договора': 'counterparty_name',
	                 'Наименование товара': 'product_name',
	                 'Ед.изм. поставщика': 'supplier_unit',
	                 'Кол-во': 'quantity',
	                 'Ед. изм.': 'unit',
	                 'Цена за единицу товара': 'unit_price',
	                 'Сумма товара': 'product_amount',
	                 'Доп. расходы': 'additional_expenses',
	                 'Общая сумма контракта по лоту': 'total_contract_amount',
	                 'Валюта контракта/договора': 'contract_currency',
	                 'Условия поставки товара': 'delivery_conditions',
	                 'Условия оплаты': 'payment_conditions',
	                 'Срок/количество дней поставки товара': 'delivery_time_days',
	                 'Дисциплина': 'discipline'}
	
	contract_df = pd.read_excel(file_path)
	contract_df = contract_df.rename(columns=dict_contract)
	df_filtered = contract_df[contract_df['contract_number'].notna()]
	
	df_filtered = df_filtered.drop_duplicates(subset=['lot_number'])  # удаляем дублирующиеся строки
	
	# в числовых полях датафрейма все NaN заменяем на нули
	df_filtered.loc[:, 'quantity'] = df_filtered.loc[:, 'quantity'].replace(np.nan, 0)
	df_filtered.loc[:, 'unit_price'] = df_filtered.loc[:, 'unit_price'].replace(np.nan, 0)
	df_filtered.loc[:, 'total_contract_amount'] = df_filtered.loc[:, 'total_contract_amount'].replace(np.nan, 0)
	df_filtered.loc[:, 'product_amount'] = df_filtered.loc[:, 'product_amount'].replace(np.nan, 0)
	df_filtered.loc[:, 'additional_expenses'] = df_filtered.loc[:, 'additional_expenses'].replace(np.nan, 0)
	
	# у executor_dak оставляем имя, фамилия, отчество. Обрезаем телефоны
	df_filtered['executor_dak'] = df_filtered['executor_dak'].apply(
		lambda x: x.partition(' (')[0] if pd.notna(x) and x != '' else x)
	
	return df_filtered


def upload_to_sql_df(df, conn, data_tmp):
	df.to_sql(data_tmp, conn, if_exists="append", index=False)
	cur = conn.cursor()
	cur.executescript(
		'''UPDATE data_tmp SET close_date = substr(close_date, 7, 4)
	                || '-' || substr(close_date, 4, 2) || '-' || substr(close_date, 1, 2);
	UPDATE data_tmp SET open_date = substr(open_date, 7, 4) || '-' || substr(open_date, 4, 2) || '-' || substr(open_date, 1, 2);''')
	
	conn.commit()


# Функция определяет номер квартала по дате
def quarter_of_date(date_tmp):
	quarter = (date_tmp.month - 1) // 3 + 1
	quarter = 'Q' + str(quarter) + '_' + date_tmp[6:10]
	return quarter


# В этом модуле идет подготовка основных укрупненных данных

def connect_to_database(db_name):
	# Эта функция собирет в список (массив) только уникальные элементы
	# из датафрейма data_df вызываем все даты закрытия лотов
	# date_strings = get_unique_only(list(data_df['close_date']))
	# получаем начальную и конечную даты
	conn = sqlite3.connect(db_name)
	cur = conn.cursor()
	min_date = cur.execute('SELECT min(close_date) FROM data_kp').fetchone()
	max_date = cur.execute('SELECT max(close_date) FROM data_kp').fetchone()
	beg_end_date = [min_date, max_date]
	
	return beg_end_date


# Метод проверки наличия файлов уже загруженных в таблицу files_names
def isfilepresent():
	list_files = []
	# подключение к базе дынных
	conn = sqlite3.connect(SQL_PATH)
	cur = conn.cursor()
	# выполнение запроса
	result = cur.execute("select nameoffiles from files_name;").fetchall()
	# получение всех результатов выборки в список
	list_files = [row[0] for row in result]
	return list_files


# добавление имени нового файла в таблицу files_name
def addfilename(file):
	# подключение к базе дынных
	conn = sqlite3.connect(SQL_PATH)
	cur = conn.cursor()
	
	# выполнение добавления
	query = "INSERT INTO files_name(nameoffiles) VALUES(?)"
	cur.execute(query, (file,))
	# сохраняем результат и закрываем базу
	conn.commit()
	conn.close()
	return
