import pandas as pd
from collections import defaultdict

def calculate_statistics(warehouse_df):
	print("Входим в метод расчета статистик по Складам")
	# Расчет количество складов за выделенный период
	# warehouses_count = warehouse_df['warehouse'].value_counts() # подсчет количества встречаемости в столбце его значений
	warehouses_unique_count = len(warehouse_df['warehouse'].unique())
	print("Количество складов = ", warehouses_unique_count)
	
	# Общее количество товарной номенклатуры на складах на последнюю дату
	max_date = warehouse_df["date_column"].max()
	print("Последняя дата --> ", max_date)
	unique_nomenclature_count = warehouse_df.loc[warehouse_df["date_column"] == max_date, "nomenclature"].nunique()
	print(f"Общее количество товарной номенклатуры на дату {max_date} = {unique_nomenclature_count}")
	
	# Считаем товарные запасы по отдельным складам общей суммой в размере валют
	# Фильтруем данные по максимальной дате
	filtered_df = warehouse_df[warehouse_df['date_column'] == max_date]
	
	# Группируем по складу и валюте, затем суммируем total_price_currency
	result_df = filtered_df.groupby(['warehouse', 'currency'])['total_price_currency'].sum().reset_index()
	
	# Выводим полученный результат в файл excel
	from excel_tables.exl_tab_warehouse_cuurrs import excel_table_warehouses_currs
	excel_table_warehouses_currs(result_df)
	