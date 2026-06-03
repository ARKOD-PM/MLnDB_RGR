import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import tensorflow as tf
from catboost import CatBoostRegressor

# Конфигурация интерфейса веб-приложения
st.set_page_config(
    page_title="РГР. Инференс моделей машинного обучения", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Настройка глобальных параметров визуализации
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14
})

# Перечень признаков в строгом соответствии с обучающей выборкой
FEATURE_COLUMNS = [
    'PT08.S1(CO)', 'C6H6(GT)', 'NOx(GT)', 'PT08.S3(NOx)', 
    'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'
]

# Статистические показатели точности моделей, полученные на тестовой выборке
METRICS_LEADERBOARD = {
    "ML1. Decision Tree Regressor": {"R2": 0.8449, "MAE": 0.3762, "Type": "Решающее дерево"},
    "ML2. Hist Gradient Boosting Regressor": {"R2": 0.9200, "MAE": 0.2599, "Type": "Градиентный бустинг (scikit-learn)"},
    "ML3. CatBoost Regressor": {"R2": 0.9282, "MAE": 0.2463, "Type": "Симметричные деревья решений (CatBoost)"},
    "ML4. Random Forest Regressor": {"R2": 0.9214, "MAE": 0.2482, "Type": "Ансамбль бэггинга"},
    "ML5. Stacking Regressor": {"R2": 0.9297, "MAE": 0.2381, "Type": "Многоуровневый стекинг"},
    "ML6. Deep Neural Network": {"R2": 0.9114, "MAE": 0.2803, "Type": "Полносвязная нейронная сеть (Keras)"}
}

@st.cache_data
def load_historical_data():
    if not os.path.exists('AirQualityUCI.csv'):
        return None
    df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')
    df.dropna(how="all", axis=1, inplace=True)
    df.dropna(how="all", axis=0, inplace=True)
    df.replace(-200, np.nan, inplace=True)
    df['Time'] = df['Time'].str.replace('.', ':', regex=False)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('DateTime', inplace=True)
    df.drop(['Date', 'Time', 'NMHC(GT)', 'PT08.S2(NMHC)'], axis=1, errors='ignore', inplace=True)
    df.interpolate(method='time', inplace=True)
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df

df_clean = load_historical_data()
scaler_path = 'models/scaler.pkl'
scaler_available = os.path.exists(scaler_path)

# Навигационная панель
st.sidebar.title("Управление проектом")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Разделы работы", 
    ["Сведения о разработчике", 
     "Анализ структуры данных (EDA)", 
     "Визуализация закономерностей",
     "Инференс прогнозных моделей"]
)

st.sidebar.markdown("---")
if scaler_available:
    st.sidebar.info("Состояние системы. Веса моделей и параметры скейлера успешно загружены.")
else:
    st.sidebar.warning("Состояние системы. Требуется предварительное обучение моделей.")

# РАЗДЕЛ 1. СВЕДЕНИЯ О РАЗРАБОТЧИКЕ
if page == "Сведения о разработчике":
    st.title("Расчетно-графическая работа")
    st.subheader("Тема. Разработка многостраничного веб-приложения для инференса ансамблевых моделей и нейронных сетей")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Идентификация автора")
        if os.path.exists("developer_photo.jpg"):
            st.image("developer_photo.jpg", caption="Фотографическая карточка разработчика", use_container_width=True)
        else:
            st.info("Для отображения фотографии разместите файл 'developer_photo.jpg' в корневом каталоге проекта.")
            
    with col2:
        st.subheader("Данные студента")
        st.markdown("""
        * **Исполнитель:** Аркашев Константин Дмитриевич 
        * **Учебная группа:** ФИТ-241
        * **Кафедра:** Прикладная математика и фундаментальная информатика 
        * **Учебное заведение:** Омский государственный технический университет (ОмГТУ)
        * **Дисциплина:** Машинное обучение и большие данные
        """)
        
        st.subheader("Программный стек проекта")
        st.markdown("""
        * Интерпретатор: Python 3.12
        * Компоненты интеллектуального анализа: Scikit-Learn, CatBoost, TensorFlow (Keras)
        * Визуализация и интерфейс: Streamlit Framework
        * Компоненты сериализации: Модуль Pickle, нативный формат Keras
        """)

# РАЗДЕЛ 2. АНАЛИЗ СТРУКТУРЫ ДАННЫХ
elif page == "Анализ структуры данных (EDA)":
    st.title("Спецификация и разведочный анализ набора данных")
    st.markdown("---")
    
    if df_clean is None:
        st.error("Файл данных 'AirQualityUCI.csv' не обнаружен в корневой директории.")
    else:
        # Текстовое описание предметной области
        st.subheader("Описание предметной области и физического смысла признаков")
        st.markdown("""
        Исследуемый массив данных содержит результаты непрерывного полуавтоматического мониторинга состава атмосферного воздуха. 
        Сбор физико-химических показателей осуществлялся в зоне промышленно развитого города Италии с помощью мультисенсорной мини-лаборатории. 
        Целевой переменной регрессионного моделирования является истинная концентрация оксида углерода CO(GT). 
        Остальные параметры представляют собой косвенные замеры сопутствующих газов и физические свойства окружающей среды.
        """)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Исходный фрагмент данных", 
            "Описательная статистика", 
            "Параметры размерности", 
            "Методология предобработки"
        ])
        
        with tab1:
            st.subheader("Выборка первых записей временного ряда")
            rows_count = st.slider("Количество отображаемых строк", 5, 50, 10)
            st.dataframe(df_clean.head(rows_count), use_container_width=True)
            
        with tab2:
            st.subheader("Математические параметры распределения признаков")
            st.dataframe(df_clean.describe().T, use_container_width=True)
            st.markdown("""
            Анализ дисперсии. Значительные расхождения в среднеквадратических отклонениях и диапазонах значений датчиков (от долей единиц для бензола до тысяч единиц для оксидов азота) математически обосновывают необходимость применения принудительной стандартизации StandardScaler. Без Z-масштабирования веса нейронной сети и коэффициенты стэкинга будут искажены в пользу признаков с большей абсолютной величиной.
            """)
            
        with tab3:
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.subheader("Типы данных и формат признаков")
                types_df = pd.DataFrame({"Тип данных (Dtype)": df_clean.dtypes.astype(str)})
                st.table(types_df)
            with col_info2:
                st.subheader("Сводные метрики массива")
                st.metric("Общее число наблюдений (строк)", df_clean.shape[0])
                st.metric("Количество независимых предикторов", df_clean.shape[1] - 1)
                st.metric("Идентификатор целевой переменной", "CO(GT)")
                
        with tab4:
            st.subheader("Этапы инженерии признаков и фильтрации шума")
            st.markdown("""
            В ходе выполнения первой лабораторной работы был реализован комплексный аналитический конвейер очистки данных.
            
            1. Исключение маркеров отсутствия информации. Сырые показатели прибора содержали аномальные значения -200, кодирующие сбой в электросети или период калибровки сенсоров. Все подобные вхождения были программно замещены на незаполненные объекты NaN.
            
            2. Фильтрация целевого vectors. Строки с неопределенным значением CO(GT) были полностью удалены из выборки. Это предотвращает обучение прогнозных алгоритмов на искусственно интерполированных ответах.
            
            3. Удаление неинформативных признаков. Столбец NMHC(GT) (концентрация неметановых углеводородов) был исключен из матрицы признаков ввиду критического объема пропусков, превышающего 90 процентов. Признак PT08.S2(NMHC) удален для предотвращения эффекта мультиколлинеарности, выявленного в ходе корреляционного анализа Пирсона.
            
            4. Восстановление пропущенных значений. Пропуски в физических предикторах были восполнены методом кусочно-линейной временной интерполяции, учитывающим непрерывную структуру метеорологических процессов.
            """)

# РАЗДЕЛ 3. ВИЗУАЛИЗАЦИЯ ЗАКОНОМЕРНОСТЕЙ
elif page == "Визуализация закономерностей":
    st.title("Графический анализ физико-химических зависимостей")
    st.markdown("---")
    
    if df_clean is None:
        st.error("Файл данных 'AirQualityUCI.csv' не обнаружен. Построение графиков невозможно.")
    else:
        st.markdown("### Сводная аналитическая панель")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("#### 1. Распределение концентрации CO(GT)")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.histplot(df_clean['CO(GT)'], bins=40, kde=True, color="steelblue", ax=ax1)
            ax1.set_xlabel("Истинная концентрация CO, мг/м³")
            ax1.set_ylabel("Частота")
            st.pyplot(fig1)
            plt.close(fig1)
            
            st.markdown("#### 2. Зависимость концентрации CO(GT) от NOx(GT)")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df_clean, x='NOx(GT)', y='CO(GT)', alpha=0.3, color="darkorange", ax=ax2)
            ax2.set_xlabel("Концентрация оксидов азота NOx, ppb")
            ax2.set_ylabel("Концентрация CO, мг/м³")
            st.pyplot(fig2)
            plt.close(fig2)
            
        with col_g2:
            st.markdown("#### 3. Анализ вариации CO(GT) по температурным режимам")
            df_plots = df_clean.copy()
            df_plots['Temp_Group'] = pd.qcut(df_plots['T'], q=4, labels=['Низкая', 'Умеренная', 'Повышенная', 'Высокая'])
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df_plots, x='Temp_Group', y='CO(GT)', palette="vlag", ax=ax3, hue='Temp_Group', legend=False)
            ax3.set_xlabel("Категория температуры окружающей среды")
            ax3.set_ylabel("Концентрация CO, мг/м³")
            st.pyplot(fig3)
            plt.close(fig3)
            
            st.markdown("#### 4. Матрица коэффициентов корреляции Пирсона")
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            sns.heatmap(df_clean.corr(), annot=True, cmap="seismic", fmt=".2f", ax=ax4, vmin=-1, vmax=1, annot_kws={"size": 7})
            ax4.tick_params(axis='both', which='major', labelsize=7)
            st.pyplot(fig4)
            plt.close(fig4)

        st.markdown("---")
        st.subheader("Аналитические выводы по результатам визуализации")
        st.markdown("""
        Распределение целевой переменной CO(GT) обладает выраженной асимметрией. 
        Диаграмма рассеяния и корреляционная матрица фиксируют сильную прямую линейную зависимость 
        между концентрацией угарного газа и оксидами азота. Значение коэффициента Пирсона превышает 0.79. 
        Диаграмма размаха указывает на стабильный уровень медианных значений выбросов в различные температурные периоды.
        """)

# РАЗДЕЛ 4. ИНФЕРЕНС ПРОГНОЗНЫХ МОДЕЛЕЙ (ИСПРАВЛЕННЫЙ БЛОК CSV)
elif page == "Инференс прогнозных моделей":
    st.title("Модуль предиктивного вывода и оценки эффективности")
    st.markdown("---")
    
    if not scaler_available:
        st.error("Критическая ошибка. Файл 'models/scaler.pkl' не найден. Выполните предварительное обучение через 'train_models.py'.")
    else:
        with open(scaler_path, 'rb') as sf:
            scaler = pickle.load(sf)
            
        selected_model = st.selectbox("Выберите математическую модель для выполнения прогноза", list(METRICS_LEADERBOARD.keys()))
        
        m_info = METRICS_LEADERBOARD[selected_model]
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Класс архитектуры алгоритма", m_info["Type"])
        with col_m2:
            st.metric("Коэффициент детерминации (R2)", f"{m_info['R2']:.4f}")
        with col_m3:
            st.metric("Средняя абсолютная ошибка (MAE)", f"{m_info['MAE']:.4f} мг/м³")
            
        st.markdown("---")
        
        input_method = st.radio("Метод подачи входных векторов параметров", 
                                ["Ручной ввод физико-химических величин", "Пакетная обработка файла таблицы (.CSV)"])
        
        def execute_pipeline_inference(model_name, data_df):
            scaled_matrix = scaler.transform(data_df[FEATURE_COLUMNS])
            
            if "ML1" in model_name:
                with open('models/ml1_classical.pkl', 'rb') as f: m = pickle.load(f)
                return m.predict(scaled_matrix)
            elif "ML2" in model_name:
                with open('models/ml2_boosting.pkl', 'rb') as f: m = pickle.load(f)
                return m.predict(scaled_matrix)
            elif "ML3" in model_name:
                m = CatBoostRegressor()
                m.load_model('models/ml3_catboost.cbm')
                return m.predict(scaled_matrix)
            elif "ML4" in model_name:
                with open('models/ml4_bagging.pkl', 'rb') as f: m = pickle.load(f)
                return m.predict(scaled_matrix)
            elif "ML5" in model_name:
                with open('models/ml5_stacking.pkl', 'rb') as f: m = pickle.load(f)
                return m.predict(scaled_matrix)
            elif "ML6" in model_name:
                m = tf.keras.models.load_model('models/ml6_nn.keras')
                return m.predict(scaled_matrix, verbose=0).flatten()
            return None

        # Режим ручного ввода замеров
        if "Ручной ввод" in input_method:
            st.subheader("Задайте текущие физико-химические показатели воздушной среды")
            
            col_in1, col_in2, col_in3 = st.columns(3)
            with col_in1:
                s1_co = st.number_input("PT08.S1(CO) (Оксид углерода) [отклик датчика]", 400.0, 2600.0, 1100.0)
                c6h6 = st.number_input("C6H6(GT) (Бензол) [мг/м³]", 0.1, 75.0, 11.5)
                nox = st.number_input("NOx(GT) (Оксиды азота) [ppb]", 1.0, 1500.0, 240.0)
            with col_in2:
                s3_nox = st.number_input("PT08.S3(NOx) (Оксиды азота) [отклик датчика]", 300.0, 2700.0, 830.0)
                no2 = st.number_input("NO2(GT) (Диоксид азота) [мг/м³]", 1.0, 340.0, 113.0)
                s4_no2 = st.number_input("PT08.S4(NO2) (Диоксид азота) [отклик датчика]", 500.0, 2900.0, 1450.0)
            with col_in3:
                s5_o3 = st.number_input("PT08.S5(O3) (Озон) [отклик датчика]", 200.0, 2600.0, 1020.0)
                temp = st.number_input("T (Температура среды) [°C]", -10.0, 50.0, 18.3)
                rh = st.number_input("RH (Относительная влажность) [%]", 5.0, 95.0, 49.2)
                ah = st.number_input("AH (Абсолютная влажность)", 0.1, 2.5, 1.02)

            raw_input_df = pd.DataFrame([[s1_co, c6h6, nox, s3_nox, no2, s4_no2, s5_o3, temp, rh, ah]], columns=FEATURE_COLUMNS)
            
            if st.button("Выполнить расчет концентрации CO"):
                prediction = execute_pipeline_inference(selected_model, raw_input_df)[0]
                st.info("Математический инференс успешно завершен.")
                st.metric(
                    label=f"Расчетная концентрация CO(GT) по версии {selected_model.split('.')[0]}", 
                    value=f"{prediction:.3f} мг/м³"
                )

        # Режим пакетной загрузки CSV таблиц (Исправленное логическое условие)
        elif "Пакетная обработка" in input_method:
            st.subheader("Пакетная предиктивная аналитика многомерных таблиц")
            st.markdown(f"Загружаемый файл должен содержать заголовки со следующей структурой признаков: `{', '.join(FEATURE_COLUMNS)}`")
            
            uploaded_file = st.file_uploader("Выберите файл таблицы формата .CSV", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    user_df = pd.read_csv(uploaded_file, sep=None, engine='python')
                    missing_cols = [c for c in FEATURE_COLUMNS if c not in user_df.columns]
                    
                    if missing_cols:
                        st.error(f"Ошибка валидации структуры файла. Отсутствуют обязательные столбцы: {missing_cols}")
                    else:
                        st.success("Структура таблицы успешно верифицирована.")
                        st.dataframe(user_df.head(5), use_container_width=True)
                        
                        if st.button("Запустить сквозной расчет по массиву"):
                            batch_predictions = execute_pipeline_inference(selected_model, user_df)
                            
                            model_id = selected_model.split(".")[0]
                            user_df[f'Прогноз_CO_({model_id})'] = np.round(batch_predictions, 4)
                            
                            st.info(f"Пакетный пересчет завершен. Количество обработанных записей: {len(user_df)}")
                            st.dataframe(user_df, use_container_width=True)
                            
                            csv_buffer = user_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Скачать результаты обработки в формате CSV",
                                data=csv_buffer,
                                file_name="AirQuality_Batch_Results.csv",
                                mime="text/csv"
                            )
                except Exception as ex:
                    st.error(f"Произошел сбой при чтении или обработке загруженного файла: {ex}")
