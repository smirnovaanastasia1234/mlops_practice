#!/bin/bash

# Запуск скрипта создания данных
python3 data_creation.py
# Запуск скрипта для предобработки данных
python3 data_preprocessing.py

# Запуск скрипта для подготовки модели
python3 model_preparation.py

# Запуск скрипта для тестирования модели и получения оценки метрики
output=$(python3 model_testing.py)

# Извлечение оценки метрики из вывода
metric=$(echo "$output" | awk '{print $NF}')

# Вывод оценки метрики в стандартный поток вывода
echo "Model test accuracy is: $metric"

# Чтение значения MSE из файла и вывод на экран
mse=$(cat test/mse.txt)
echo "Mean Squared Error is: $mse"