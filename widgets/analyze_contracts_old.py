import pandas as pd
import os
import gc
import networkx as nx
import matplotlib.pyplot as plt


def analyze_discrepancies(filtered_df):
	print('Мы в методе')
	results = []
	for _, row in filtered_df.iterrows():
		lot_number = row['lot_number']
		supplier_kp = row['counterparty_name']
		total_price_kp = row['total_price_kp']
		total_contract_amount_contract = row['total_contract_amount_contract']
		supplier_qty_kp = row['supplier_qty_kp']
		quantity_contract = row['quantity_contract']
		unit_price_kp = row['unit_price_kp']
		unit_price_contract = row['unit_price_contract']
		
		# Анализ изменений параметров
		if supplier_qty_kp == quantity_contract and unit_price_kp == unit_price_contract:
			# Параметры не изменились
			results.append({
				'lot_number': lot_number,
				'supplier': supplier_kp,
				'status': 'Изменение суммы без изменения параметров. Возможно искусственное изменение.',
				'difference': total_contract_amount_contract - total_price_kp
			})
		else:
			# Параметры изменились, нужно оценить ситуацию
			if (quantity_contract * unit_price_contract) > (supplier_qty_kp * unit_price_kp):
				results.append({
					'lot_number': lot_number,
					'supplier': supplier_kp,
					'status': 'Изменение параметров в пользу компании.',
					'difference': total_contract_amount_contract - total_price_kp
				})
			else:
				results.append({
					'lot_number': lot_number,
					'supplier': supplier_kp,
					'status': 'Изменение параметров в ущерб компании.',
					'difference': total_contract_amount_contract - total_price_kp
				})
	return pd.DataFrame(results)


def analyzeNonEquilSums(data_df, contract_df):
	# Агрегация данных по лотам в data_df
	kp_agg = data_df.groupby(['lot_number', 'currency', 'winner_name']).agg(
		total_price_kp=pd.NamedAgg(column='total_price', aggfunc='sum'),
		supplier_qty_kp=pd.NamedAgg(column='supplier_qty', aggfunc='mean'),
		unit_price_kp=pd.NamedAgg(column='unit_price', aggfunc='mean'),
		close_date_kp=pd.NamedAgg(column='close_date', aggfunc='max')
	).reset_index()
	
	# Агрегация данных по контрактам в contract_df
	contract_agg = contract_df.groupby(['lot_number', 'contract_currency', 'counterparty_name']).agg(
		total_contract_amount_contract=pd.NamedAgg(column='total_contract_amount', aggfunc='sum'),
		quantity_contract=pd.NamedAgg(column='quantity', aggfunc='sum'),
		unit_price_contract=pd.NamedAgg(column='unit_price', aggfunc='mean'),
		contract_signing_date=pd.NamedAgg(column='contract_signing_date', aggfunc='max')
	).reset_index()
	
	# Соединение двух датафреймов на основе lot_number, currency и winner_name/counterparty_name
	merged_df = kp_agg.merge(
		contract_agg,
		left_on=['lot_number', 'currency', 'winner_name'],
		right_on=['lot_number', 'contract_currency', 'counterparty_name'],
		suffixes=('_kp', '_contract')
	)
	
	# Фильтрация строк, где разница между суммой лота и контракта больше 0.01
	filtered_df = merged_df[abs(merged_df['total_price_kp'] - merged_df['total_contract_amount_contract']) > 0.01]
	
	print(filtered_df[['lot_number', 'currency', 'total_price_kp', 'total_contract_amount_contract']])
	
	analysis_results = analyze_discrepancies(filtered_df)

	
	print(analysis_results[['lot_number', 'status']])
	
	