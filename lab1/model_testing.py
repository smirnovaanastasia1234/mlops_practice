import os
import pickle
import pandas as pd

# Загрузка сохраненной модели из файла
model_file_path = 'model.pkl'
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)

# Загрузка данных для тестирования из файла
test_data_path = 'test/test_data.csv'
test_data = pd.read_csv(test_data_path)

# Преобразование столбца 'Date' в тип данных datetime
test_data['Date'] = pd.to_datetime(test_data['Date'])

# Извлечение числовых характеристик из столбца 'Date'
test_data['Year'] = test_data['Date'].dt.year
test_data['Month'] = test_data['Date'].dt.month
test_data['Day'] = test_data['Date'].dt.day

# Выделение признаков для тестирования
X_test = test_data[['Year', 'Month', 'Day']]
y_test = test_data['Temperature']

# Применение модели на тестовых данных
predictions = model.predict(X_test)

# Вычисление и вывод метрик для оценки модели (например, среднеквадратичная ошибка)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)
# Сохранение значения MSE в файл
mse_file_path = 'test/mse.txt'
with open(mse_file_path, 'w') as f:
    f.write(str(mse))