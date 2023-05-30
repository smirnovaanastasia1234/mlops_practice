import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Загрузка данных из файла
train_data_path = 'train/train_data.csv'
train_data = pd.read_csv(train_data_path)

# Преобразование столбца 'Date' в тип данных datetime
train_data['Date'] = pd.to_datetime(train_data['Date'])

# Извлечение числовых характеристик из столбца 'Date'
train_data['Year'] = train_data['Date'].dt.year
train_data['Month'] = train_data['Date'].dt.month
train_data['Day'] = train_data['Date'].dt.day

# Разделение данных на признаки (X) и целевую переменную (y)
X = train_data[['Year', 'Month', 'Day']]
y = train_data['Temperature']

# Разделение данных на обучающий и проверочный наборы
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Сохранение модели в файл
model_file_path = 'model.pkl'
with open(model_file_path, 'wb') as f:
    pickle.dump(model, f)