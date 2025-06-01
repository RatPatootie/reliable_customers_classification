import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
# custom files
import model_best_hyperparameters
import columns
import warnings
import sys 
warnings.simplefilter('ignore')

# read train data
#ds = pd.read_csv("train_data.csv")
try:
    ds = pd.read_csv(r"C:\Users\Петро\OneDrive\Робочий стіл\Labs\Basic of smart technology and system\Lar_5\lab_5_project_Бодруг\data\train.csv")
    print("Не помилка: Файл з новими даними  знайдено.")
except FileNotFoundError:
    print("Помилка: Файл з новими даними не знайдено.")
    sys.exit(1)
# feature engineering
# Missing data imputation

def impute_na(df, variable, value):
    return df[variable].fillna(value)

# Let's create a dict and impute mean values
mean_impute_values = dict()
for column in columns.mean_impute_columns:
    mean_impute_values[column] = ds[column].mean()
    ds[column] = impute_na(ds, column, mean_impute_values[column])

# Let's create a dict and impute mode values
mode_impute_values = dict()
for column in columns.mode_impute_columns:
    mode_impute_values[column] = ds[column].mode()[0]
    ds[column] = impute_na(ds, column, mode_impute_values[column])

# Outlier Engineering
def find_skewed_boundaries(df, variable, distance):
    df[variable] = pd.to_numeric(df[variable],errors='coerce')
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary

upper_lower_limits = dict()
for column in columns.outlier_columns:
    upper_lower_limits[column+'_upper_limit'], upper_lower_limits[column+'_lower_limit'] = find_skewed_boundaries(ds, column, 5)
for column in columns.outlier_columns:
    ds = ds[~ np.where(ds[column] > upper_lower_limits[column+'_upper_limit'], True,
                       np.where(ds[column] < upper_lower_limits[column+'_lower_limit'], True, False))]

# Categorical encoding
map_dicts = dict()
for column in columns.cat_columns:
    ds[column] = ds[column].astype('category')
    map_dicts[column] = dict(zip(ds[column], ds[column].cat.codes))
    ds[column] = ds[column].cat.codes

#####
encoded_dicts = dict()

for column in columns.hot_columns:
    # Виконуємо one-hot encoding для поточного стовпця
    dummies = pd.get_dummies(ds[column], prefix=column)

    # Зберігаємо список нових стовпців у словник
    encoded_dicts[column] = dummies.columns.tolist()

    # Додаємо нові закодовані стовпці до датасету
    ds = pd.concat([ds, dummies], axis=1)

    # Видаляємо оригінальний категоріальний стовпець
    ds.drop(column, axis=1, inplace=True)
bool_cols = ds.select_dtypes(include='bool').columns
ds[bool_cols] = ds[bool_cols].astype(int)

#нормалізація
scaler = MinMaxScaler()
ds[columns.normis] = scaler.fit_transform(ds[columns.normis])

# Збереження скейлера у файл
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f) 
     
# save parameters 
param_dict = {'mean_impute_values':mean_impute_values,
              'mode_impute_values':mode_impute_values,
              'upper_lower_limits':upper_lower_limits,
              'map_dicts':map_dicts,
              'encoded_dicts':encoded_dicts
             }
with open('param_dict.pickle', 'wb') as handle:
    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Define target and features columns
X = ds[columns.X_columns]
y = ds[columns.y_column]

# Let's say we want to split the data in 90:10 for train:test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.9)

#smote
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train) 

# Building and train knn model
kn = KNeighborsClassifier(**model_best_hyperparameters.params)
kn.fit(X_train_resampled, y_train_resampled)
y_pred = kn.predict(X_test)

# Print metrics
print('Test set metrics:\n', metrics.classification_report(y_test, y_pred))
report = metrics.classification_report(y_test, y_pred)
with open('../docs/classification_report.txt', 'w') as f:
    f.write(report)

print('Звіт про метрики збережено у файл classification_report.txt')

# Permutation importance
result = permutation_importance(kn, X_test[:300], y_test[:300], n_repeats=10, random_state=42)

# Create a DataFrame for importance results
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': result.importances_mean})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Save feature importance to a CSV file
importance_df.to_csv('../docs/feature_importance.csv', index=False)

filename = '../models/finalized_model.sav'
pickle.dump(kn, open(filename, 'wb'))
print('fenite')