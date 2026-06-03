import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
import tensorflow as tf

# Создание директории для весов моделей
os.makedirs('models', exist_ok=True)

print("1. Загрузка и очистка данных AirQualityUCI по правилам ЛР №1...")
df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')
df.dropna(how="all", axis=1, inplace=True)
df.dropna(how="all", axis=0, inplace=True)

# Замена маркеров пропусков -200 на NaN
df.replace(-200, np.nan, inplace=True)

# Форматирование временных индексов
df['Time'] = df['Time'].str.replace('.', ':', regex=False)
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('DateTime', inplace=True)
df.drop(['Date', 'Time', 'NMHC(GT)'], axis=1, errors='ignore', inplace=True)

# Исключаем строки, где отсутствует целевой признак CO(GT), чтобы не переобучаться на интерполяции ответов
df.dropna(subset=['CO(GT)'], inplace=True)

# Заполнение пропусков в признаках методом временной интерполяции
df.interpolate(method='time', inplace=True)
df.bfill(inplace=True)
df.ffill(inplace=True)

# Удаление мультиколлинеарного признака PT08.S2
df.drop('PT08.S2(NMHC)', axis=1, errors='ignore', inplace=True)

# Разделение на предикторы и таргет
X = df.drop(columns=['CO(GT)'])
y = df['CO(GT)']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Стандартизация признаков (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Сохраняем скейлер для app.py
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("3. Обучение моделей на основе ваших лабораторных работ...")

# ML1: Ваше дерево решений из ЛР4
ml1 = DecisionTreeRegressor(max_depth=4, random_state=42)
ml1.fit(X_train_scaled, y_train)

# ML2: Ваш гистограммный бустинг из ЛР4
ml2 = HistGradientBoostingRegressor(random_state=42)
ml2.fit(X_train_scaled, y_train)

# ML3: Ваш CatBoost из ЛР4
ml3 = CatBoostRegressor(verbose=0, random_state=42, thread_count=-1)
ml3.fit(X_train_scaled, y_train)

# ML4: Ваш Random Forest из ЛР4
ml4 = RandomForestRegressor(random_state=42, n_jobs=-1)
ml4.fit(X_train_scaled, y_train)

# ML5: Ваша точная топовая архитектура StackingRegressor из lab4.ju.py
stack_estimators = [
    ('rf', RandomForestRegressor(random_state=42, n_jobs=-1)),
    ('gb', HistGradientBoostingRegressor(random_state=42)),
    ('knn', KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
    ('svm', LinearSVR(random_state=42, dual="auto", max_iter=2000))
]
ml5 = StackingRegressor(estimators=stack_estimators, final_estimator=Ridge())
ml5.fit(X_train_scaled, y_train)

# ML6: Полносвязная нейросеть Keras, адаптированная под масштабированные данные РГР
ml6 = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
ml6.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')
ml6.fit(X_train_scaled, y_train, epochs=30, batch_size=64, verbose=0)

print("\n📈 ПРОВЕРКА КАЧЕСТВА НА ТЕСТОВОЙ ВЫБОРКЕ (РЕАЛЬНЫЕ МЕТРИКИ):")
models_dict = {
    "ML1 (Decision Tree)": ml1, 
    "ML2 (HistGB)": ml2, 
    "ML4 (Random Forest)": ml4, 
    "ML5 (Ваш Стэкинг)": ml5
}

for name, model in models_dict.items():
    preds = model.predict(X_test_scaled)
    print(f" -> {name}: R2 = {r2_score(y_test, preds):.4f} | MAE = {mean_absolute_error(y_test, preds):.4f}")

# Оценка CatBoost
preds_cb = ml3.predict(X_test_scaled)
print(f" -> ML3 (CatBoost): R2 = {r2_score(y_test, preds_cb):.4f} | MAE = {mean_absolute_error(y_test, preds_cb):.4f}")

# Оценка Keras
preds_nn = ml6.predict(X_test_scaled, verbose=0).flatten()
print(f" -> ML6 (Neural Network): R2 = {r2_score(y_test, preds_nn):.4f} | MAE = {mean_absolute_error(y_test, preds_nn):.4f}")

print("\n4. Сериализация моделей...")
with open('models/ml1_classical.pkl', 'wb') as f: pickle.dump(ml1, f)
with open('models/ml2_boosting.pkl', 'wb') as f: pickle.dump(ml2, f)
ml3.save_model('models/ml3_catboost.cbm')
with open('models/ml4_bagging.pkl', 'wb') as f: pickle.dump(ml4, f)
with open('models/ml5_stacking.pkl', 'wb') as f: pickle.dump(ml5, f)
ml6.save('models/ml6_nn.keras')

print("Все модели успешно обучены по вашим мета-параметрам и сохранены!")
