import pandas as pd


def unique_discip_actor_lots(df):
	unique_disciplines = df['discipline'].unique()
	lots_per_actor = {}
	
	for discipline in unique_disciplines:
		df_filtered = df[df['discipline'] == discipline].copy()
		if not df_filtered.empty:
			# Группируем по исполнителю и считаем количество лотов для каждого победителя
			lots_info = df_filtered.groupby(['actor_name', 'winner_name'])['lot_number'].count().unstack(
				fill_value=0).to_dict(orient='index')
			
			# Убираем нулевые значения (если нужно)
			lots_info_cleaned = {
				actor: {winner: count for winner, count in winners.items() if count > 0}
				for actor, winners in lots_info.items()
			}
			
			lots_per_actor[discipline] = lots_info_cleaned
		else:
			lots_per_actor[discipline] = {}
	
	return lots_per_actor


def analyze_suppliers(lots_per_actor):
	import pandas as pd
	from sklearn.cluster import KMeans
	from sklearn.preprocessing import StandardScaler
	from sklearn.decomposition import PCA
	import matplotlib.pyplot as plt
	
	# Создаем supplier_stats
	supplier_stats = {}
	for discipline, actors in lots_per_actor.items():
		supplier_stats[discipline] = {}
		for actor, suppliers in actors.items():
			supplier_stats[discipline][actor] = suppliers.copy()
	
	# Создаем список всех уникальных поставщиков
	all_suppliers = set()
	for discipline, actors in supplier_stats.items():
		for actor, suppliers in actors.items():
			all_suppliers.update(suppliers.keys())
	all_suppliers = list(all_suppliers)
	
	# создаем DataFrame
	data = []
	for discipline, actors in supplier_stats.items():
		for actor, suppliers in actors.items():
			row = {supplier: suppliers.get(supplier, 0) for supplier in all_suppliers}
			row['discipline'] = discipline
			row['actor'] = actor
			data.append(row)
	
	df = pd.DataFrame(data)
	
	# Устанавливаем индекс (исполнитель + дисциплина)
	df.set_index(['discipline', 'actor'], inplace=True)
	
	# Нормализация данных
	scaler = StandardScaler()
	df_scaled = scaler.fit_transform(df)
	
	# Кластеризация K-Means
	kmeans = KMeans(n_clusters=3, random_state=42)
	df['cluster'] = kmeans.fit_predict(df_scaled)
	
	# Визуализация с использованием PCA
	pca = PCA(n_components=2)
	df_pca = pca.fit_transform(df_scaled)
	df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'], index=df.index)
	
	plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df['cluster'], cmap='viridis', s=100)
	plt.xlabel('Principal Component 1')
	plt.ylabel('Principal Component 2')
	plt.title('Кластеризация исполнителей (PCA)')
	plt.show()

