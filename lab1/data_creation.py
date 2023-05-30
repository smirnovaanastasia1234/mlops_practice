import os
import numpy as np
import pandas as pd

# Создание папок "train" и "test", если они не существуют
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

# Функция для создания набора данных с аномалиями
def create_dataset_with_anomalies(start_date, end_date, mean_temp, anomaly_dates, anomaly_values, noise_std):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    temperatures = np.random.normal(loc=mean_temp, scale=noise_std, size=len(dates))
    
    # Добавление аномалий
    for anomaly_date, anomaly_value in zip(anomaly_dates, anomaly_values):
        index = dates.get_loc(anomaly_date)
        temperatures[index] += anomaly_value
    
    return pd.DataFrame({'Date': dates, 'Temperature': temperatures})

# Параметры для создания набора данных
start_date = '2023-01-01'
end_date = '2023-12-31'
mean_temp = 25.0
anomaly_dates = ['2023-02-15', '2023-06-30', '2023-09-10']
anomaly_values = [10.0, -5.0, 8.0]
noise_std = 1.0

# Создание тренировочного набора данных
train_dataset = create_dataset_with_anomalies(start_date, end_date, mean_temp, anomaly_dates, anomaly_values, noise_std)
train_dataset.to_csv('train/train_data.csv', index=False)

# Создание тестового набора данных
test_dataset = create_dataset_with_anomalies(start_date, end_date, mean_temp, anomaly_dates, anomaly_values, noise_std)
test_dataset.to_csv('test/test_data.csv', index=False)