import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загрузка данных из файлов
train_data_path = 'train/train_data.csv'
test_data_path = 'test/test_data.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Извлечение признаков и целевой переменной из обучающей выборки
X_train = train_data.drop('Temperature', axis=1)
y_train = train_data['Temperature']

# Извлечение признаков и целевой переменной из тестовой выборки
X_test = test_data.drop('Temperature', axis=1)
y_test = test_data['Temperature']

# Преобразование столбца Date в тип данных datetime
X_train['Date'] = pd.to_datetime(X_train['Date'])
X_test['Date'] = pd.to_datetime(X_test['Date'])

# Применение предобработки данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Сохранение предобработанных данных в новых файлах
preprocessed_train_data_path = 'train/train_data_scaled.csv'
preprocessed_test_data_path = 'test/test_data_scaled.csv'
train_data_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train_data_scaled['Temperature'] = y_train
train_data_scaled.to_csv(preprocessed_train_data_path, index=False)
test_data_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_data_scaled['Temperature'] = y_test
test_data_scaled.to_csv(preprocessed_test_data_path, index=False)