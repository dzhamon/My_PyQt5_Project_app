from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Регистрация шрифта
def register_fonts():
    try:
        pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
        logging.info("Шрифт DejaVuSans зарегистрирован успешно.")
    except Exception as e:
        logging.error(f"Ошибка при регистрации шрифта: {e}")

register_fonts()

def export_to_excel(df, save_path):
    """
    Экспортирует DataFrame в Excel.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_excel(save_path, index=False)
        logging.info(f"Данные успешно экспортированы в {save_path}")
    except Exception as e:
        logging.error(f"Ошибка при экспорте данных в Excel: {e}")
        
def save_plot(plot_func, save_path, **kwargs):
    """
    Сохраняет график, вызывая переданную функцию построения.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_func(**kwargs)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"График сохранен: {save_path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении графика: {e}")


class SeedKMeansClustering:
    def __init__(self, kpi_analyzer):
        """
        Инициализация с экземпляром MyLotAnalyzeKPI.
        """
        self.kpi_analyzer = kpi_analyzer
        self.full_data_with_clusters = None

    def perform_clustering(self):
        """
        Выполняет кластеризацию с использованием Seed Points.
        """
        try:
            # Шаг 1: Рассчитать KPI
            df_kpi = self.kpi_analyzer.calculate_kpi(self.kpi_analyzer.df)
    
            # Шаг 2: Получить Seed Data и выборку обучения
            seed_data = self.kpi_analyzer.get_seed_data(df_kpi)
            remaining_data = df_kpi[~df_kpi.index.isin(seed_data.index)].copy()  # Создаем копию!
    
    
            # Шаг 3: Подготовить данные
            seed_points = seed_data[['total_lots', 'avg_time_to_close', 'avg_lot_value', 'sum_lot_value']].values
            n_clusters = len(seed_data['discipline'].unique())
            features = remaining_data[['total_lots', 'avg_time_to_close', 'avg_lot_value', 'sum_lot_value']].values
            
            # Кластеризация
            kmeans = KMeans(n_clusters=n_clusters, init=seed_points, n_init=10)
            kmeans.fit(features)
    
            # Добавить метки кластеров к данным
            remaining_data['cluster_label'] = kmeans.labels_
            seed_data['cluster_label'] = range(len(seed_data))  # Метки кластеров для Seed Data
    
            # Шаг 4: Оценка результатов кластеризации
            silhouette_avg = silhouette_score(features, kmeans.labels_)
            logging.info(f"Silhouette Score: {silhouette_avg}")
    
            # Шаг 5: Объединить данные
            self.full_data_with_clusters = pd.concat([seed_data, remaining_data])
            return self.full_data_with_clusters, kmeans
        except Exception as error:
            logging.error(f"Ошибка кластеризации: {error}")
            return None, None
        

    def plot_cluster_distribution(self, df_clusters, save_path):
        print('Мы в Гистограмме?')
        """
        Строит гистограмму распределения по кластерам.
        :param df_clusters: DataFrame с данными кластеризации, содержащий столбец 'cluster_label'.
        """
        def plot_histogram():
            # Получение распределения
            cluster_distribution = df_clusters['cluster_label'].value_counts()
            cluster_distribution.sort_index().plot(kind='bar', color='skyblue')
            plt.title('Распределение по кластерам')
            plt.xlabel('Кластеры')
            plt.ylabel('Количество исполнителей')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        save_plot(plot_histogram, save_path)

        
   