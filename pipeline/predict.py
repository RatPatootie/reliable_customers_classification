import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys 
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# custom files
import columns

# Спроба зчитати нові дані
try:
    ds = pd.read_csv("data/new_input.csv")
except FileNotFoundError:
    print("Помилка: Файл з новими даними не знайдено.")
    sys.exit(1)

# Перевірка на порожній набір даних
if ds.empty:
    print("Попередження: Вхідний набір даних порожній. Завершення виконання.")
    sys.exit(1)

print('Розмір нових даних:', ds.shape)

# Завантаження параметрів для feature engineering
try:
    param_dict = pickle.load(open('pipeline/param_dict.pickle', 'rb'))
except FileNotFoundError:
    print("Помилка: Файл 'param_dict.pickle' не знайдено.")
    sys.exit(1)

# Missing data imputation
def impute_na(df, variable, value):
    #df[variable] = df[variable].replace(np.NaN, value)
    #return df[variable].fillna(value)
    if variable in df.columns:
        df[variable] = df[variable].fillna(value)
    else:
        print(f"Warning: '{variable}' not found in DataFrame.")
    return df[variable]
# Let's read a dict and impute mean values
for column in columns.mean_impute_columns:
    ds[column] = impute_na(ds, column, param_dict['mean_impute_values'][column])

# Let's read a dict and impute mode values
for column in columns.mode_impute_columns:
    ds[column] = impute_na(ds, column, param_dict['mode_impute_values'][column])


# Categorical encoding
for column in columns.cat_columns:
    ds[column] = ds[column].map(param_dict['map_dicts'][column])
    # to encoding missing (new) categories
    ds[column] = impute_na(ds, column, max(param_dict['map_dicts'][column].values())+1)

###`Categorical encoding`
for column in columns.hot_columns:
        # Отримуємо список імен закодованих стовпців із параметрів
        encoded_columns = param_dict['encoded_dicts'][column]

        # Створюємо порожній DataFrame для зберігання закодованих значень
        dummies = pd.DataFrame(0, index=ds.index, columns=encoded_columns)

        # Заповнюємо закодовані стовпці на основі наявних значень у датасеті
        for idx, value in ds[column].items():
            dummy_col = f"{column}_{value}"
            if dummy_col in dummies.columns:
                dummies.at[idx, dummy_col] = 1  # Встановлюємо 1 для відповідного стовпця

        # Додаємо закодовані стовпці до датасету
        ds = pd.concat([ds, dummies], axis=1)

        # Видаляємо оригінальний категоріальний стовпець
        ds.drop(column, axis=1, inplace=True)

#normis
# Завантаження скейлера
with open('pipeline/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
try:
    ds[columns.normis] = scaler.fit_transform(ds[columns.normis])
except KeyError as e:
    print(f"Попередження: {e}. Пропуск нормалізації.")

# Define target and features columns
X = ds[columns.X_columns]

# load the model and predict
try:
    kn = pickle.load(open('models/finalized_model.sav', 'rb'))
except FileNotFoundError:
    print("Помилка: Модель 'finalized_model.sav' не знайдено.")
    sys.exit(1)

try:
    y_pred = kn.predict(X)
    ds['Status'] = y_pred
except ValueError as e:
    print(f"Помилка під час передбачення: {e}")
    sys.exit(1)

# Збереження результатів
try:
    ds.to_csv('data/prediction_results.csv', index=False)
    print("Результати передбачень збережено у файл 'prediction_results.csv'.")
except Exception as e:
    print(f"Помилка під час збереження результатів: {e}")
