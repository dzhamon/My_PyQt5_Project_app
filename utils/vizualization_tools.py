import os
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def plot_bar_chart(x, y, title, x_label, y_label, output_file):
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, alpha=0.7)
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
def save_top_suppliers_bar_chart(top_suppliers, currency, interval_text, output_dir):
    """
    Создание и сохранение графиков топ-10 поставщиков
    """
    # Create a figure for the chart
    fig, ax = plt.subplots(figsize=(12, 8))
    top_suppliers.plot(kind='bar', color='skyblue', ax=ax)
    
    # Обновляем метки оси X
    ax.set_xticks(range(len(top_suppliers)))
    ax.set_xticklabels(top_suppliers.index, rotation=45)

    # Add title and labels
    ax.set_title(f'Top-10 Suppliers by Total Costs for {interval_text} (Currency: {currency})')
    ax.set_xlabel('Supplier')
    ax.set_ylabel(f'Total Costs ({currency})')

    # Add value labels on the bars
    for i, v in enumerate(top_suppliers):
        ax.text(i, v + 0.07 * v, f'{v:,.0f}', ha='center', va='bottom')

    ax.grid(axis='y')

    # Save the chart to the specified directory
    file_path = os.path.join(output_dir, f'top_suppliers_{currency}.png')
    plt.savefig(file_path)
    plt.close()
    gc.collect()


def visualize_price_differences(df):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Создаем папку для сохранения графиков, если её нет
    output_folder = r"D:\Analysis-Results\suppliers_between_disciplines"
    os.makedirs(output_folder, exist_ok=True)
    
    unique_disciplines = df['discipline1'].unique()
    for discipline in unique_disciplines:
        filtered_df = df[df['discipline1'] == discipline]
        materials = filtered_df['good_name']
        prices_discipline1 = filtered_df['price_discipline1']
        prices_discipline2 = filtered_df['price_discipline2']
        
        x = np.arange(len(materials))  # Positions for bars
        bar_width = 0.35  # ширина каждого столбца
        
        # создаем график
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bar1 = ax.bar(
            x - bar_width / 2,
            prices_discipline1,
            width=bar_width,
            label='Discipline 1',
            color='skyblue'
        )
        bar2 = ax.bar(
            x + bar_width / 2,
            prices_discipline2,
            width=bar_width,
            label='Discipline 2',
            color='orange'
        )
        
        ax.set_xlabel('Materials', fontsize=12)
        ax.set_ylabel('Unit Price', fontsize=12)
        ax.set_title(f'Comparison of Unit Prices for Discipline: {discipline}', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(materials, rotation=45, ha='right', fontsize=10)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.legend()
        
        plt.tight_layout()
        # Сохраняем график
        output_file = os.path.join(output_folder, f"{discipline}_price_comparison.png")
        plt.savefig(output_file, dpi=300)
        plt.close(fig)  # Закрываем график, чтобы не занимать память
    
    print(f"Графики успешно сохранены в папку: {output_folder}")
        
    return


def heatmap_common_suppliers(common_suppliers_df):
    print('Запускается метод heatmap_common_suppliers')
    print(common_suppliers_df.columns)
    
    # Папка для сохранения тепловой карты
    output_folder = r"D:\Analysis-Results\suppliers_between_disciplines"
    os.makedirs(output_folder, exist_ok=True)  # Создаем папку, если она не существует
    
    # Создаем матрицу пересечений
    disciplines = set(common_suppliers_df['discipline1']).union(set(common_suppliers_df['discipline2']))
    heatmap_data = pd.DataFrame(index=list(disciplines), columns=list(disciplines)).fillna(0)
    
    for _, row in common_suppliers_df.iterrows():
        discip1, discip2 = row['discipline1'], row['discipline2']
        heatmap_data.at[discip1, discip2] = len(row['common_suppliers'])
        heatmap_data.at[discip2, discip1] = len(row['common_suppliers'])
    
    # Визуализация тепловой карты
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="YlGnBu",
        fmt="g",
        cbar=True,
        square=True,
        linewidths=0.5
    )
    
    # Добавляем отступы
    plt.title("Количество общих поставщиков между дисциплинами", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout(pad=2.0)  # Устанавливаем дополнительные отступы
    
    # Сохраняем тепловую карту
    output_file = os.path.join(output_folder, "heatmap_common_suppliers.png")
    plt.savefig(output_file, dpi=300)
    plt.close()  # Закрываем график, чтобы не занимать память
    
    print(f"Тепловая карта успешно сохранена в файл: {output_file}")


def visualize_isolation_forest(analyzed_df):
    # Тепловая карта Isolation Forest
    if 'unit_price_in_eur' not in analyzed_df.columns or 'total_price_in_eur' not in analyzed_df.columns:
        print("Для визуализации нужны столбцы 'unit_price_in_eur' и 'total_price_in_eur'")
        return
    
    # Используем только два признака для 2D визуализации
    data = analyzed_df[['unit_price_in_eur', 'total_price_in_eur']].dropna()
    
    # Создаём модель Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(data)
    data['is_anomaly'] = model.predict(data)  # 1 - нормальные данные, -1 - аномалии
    
    # Создаём сетку для визуализации границ
    xx, yy = np.meshgrid(
        np.linspace(data['unit_price_in_eur'].min(), data['unit_price_in_eur'].max(), 100),
        np.linspace(data['total_price_in_eur'].min(), data['total_price_in_eur'].max(), 100)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    scores = model.decision_function(grid_points).reshape(xx.shape)
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    
    # Границы разделения
    plt.contourf(xx, yy, scores, levels=50, cmap=plt.cm.RdYlBu_r, alpha=0.6)
    
    # Точки данных
    plt.scatter(
        data['unit_price_in_eur'], data['total_price_in_eur'],
        c=data['is_anomaly'].map({1: 'blue', -1: 'red'}),
        edgecolors='k', alpha=0.8, label='Data Points'
    )
    
    plt.title('Isolation Forest - Visualization of Anomalies')
    plt.xlabel('Unit Price (EUR)')
    plt.ylabel('Total Price (EUR)')
    plt.colorbar(label='Anomaly Score')
    plt.legend(['Normal', 'Anomalies'], loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()