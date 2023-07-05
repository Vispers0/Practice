import pandas as pd
import numpy as np
import datetime as dt

import plotly.express as px
import plotly.figure_factory as ff
from pandas import Timestamp

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

import xgboost as xgb

import io

import const

df_grouped = None

df = pd.read_csv("./res/solar_weather.csv")
df = df.drop(['hour', 'month'], axis=1)
df['Time'] = pd.to_datetime(df['Time'])
df = df.set_index('Time')


# Запись датасета в файл
def write_df():
    with open(const.PATH_TXT + const.DF, 'w',
              encoding='utf-8') as file:
        file.write(df.to_string())


# Запись информации о датасете в файл
def write_info():
    buf = io.StringIO()
    df.info(buf=buf)
    with open(const.PATH_TXT + const.INFO, 'w',
              encoding='utf-8') as file:
        file.write(buf.getvalue())
        file.write("amount of rows and columns: " + str(df.shape))
        file.write("\namount of null cells:\n" + str(df.isnull().sum()))
        file.write("\namount of duplicated values: " + str(df.duplicated().sum()))


# !!!DEPRECATED!!!
# Создание нового столбца для группировки
# def make_column():
#     df['Time'] = pd.to_datetime(df['Time'])
#     df['year'] = df['Time'].dt.year


# Запись описательной статистики в файл
def write_desc():
    with open(const.PATH_TXT + const.DESC, 'w',
              encoding='utf-8') as file:
        file.write(df.describe().round(3).to_string())


# Группировка датасета по году, месяцу и часу
def group_df():
    global df_grouped
    df_grouped = df.groupby([df.index.month]).mean(numeric_only=True).round(3)

    df_grouped['isSun'] = df_grouped['isSun'].astype(float).round(0)
    df_grouped['weather_type'] = df_grouped['weather_type'].astype(float).round(0)

    with open(const.PATH_TXT + const.GROUP, 'w',
              encoding='utf-8') as file:
        file.write(df_grouped.to_string())


# Построение матрицы корреляции
def build_matrix():
    df_corr = df.corr()

    x = list(df_corr.columns)
    y = list(df_corr.index)
    z = np.array(df_corr)

    figure = ff.create_annotated_heatmap(x=x,
                                         y=y,
                                         z=z,
                                         annotation_text=np.around(z, 2))

    figure.layout.width = 1000
    figure.layout.height = 1000

    figure.write_image(const.PATH_IMG + "corr_matrix.svg")


# Построение графка общего потребления энергии
def build_general_plot():
    figure = px.line(df,
                     x=df.index,
                     y='Energy delta[Wh]',
                     title='Усреднённое ежемесячное потребление энергии с 2017 по 2022 год')

    figure.write_image(const.PATH_IMG + "general_plot.svg")


# Построение "Ящика с усами", изображающего потребление энергии ежемесячно на протяжении 5 лет
def build_box_plot():
    figure = px.box(
        df,
        x=df.index.month,
        y='Energy delta[Wh]',
        title='Ежемесячное потребление электроэнергии',
        labels={'x': 'Months'}
    )

    figure.update_traces(width=0.5)
    figure.write_image(const.PATH_IMG + 'box_plot.svg')


# Построение гистограмм потребления энергии и факторов, влияющих на него
def build_general_bar():
    figure = px.bar(df_grouped,
                    x=df_grouped.index,
                    y='Energy delta[Wh]',
                    labels={'x':'Months'},
                    title='Ежемесячное потребление электроэнергии',
                    color='Energy delta[Wh]')

    figure.update_layout(barmode='group')
    figure.write_image(const.PATH_IMG + "general_bar_plot.svg")


def build_ghi_bar():
    figure = px.bar(df_grouped,
                    x=df_grouped.index,
                    y='GHI',
                    labels={'x':'Months'},
                    title='Ежемесячное глобольное горизонтальное излучение',
                    color='GHI')

    figure.update_layout(barmode='group')
    figure.write_image(const.PATH_IMG + "GHI_bar_plot.svg")


# Построение "ящика с усами" для общего потребления энергии и GHI
def build_consumption_box():
    figure = px.box(df_grouped,
                    y='Energy delta[Wh]',
                    title='Общее потребление электроэнергии')

    figure.write_image(const.PATH_IMG + 'consumption_box_plot.svg')


def build_ghi_box():
    figure = px.box(df_grouped,
                    y='GHI',
                    title='Общее глобальное горизонтальное излучение')

    figure.write_image(const.PATH_IMG + 'GHI_box_plot.svg')


# Построение графика зависимости потребления электроэнергии от глобального горизонтального излучения
def build_cons_ghi_scatter():
    figure = px.bar(df_grouped,
                        x='GHI',
                        y='Energy delta[Wh]',
                        title='Зависимость потребления электроэнергии от глобального горизонтального излучения')

    figure.update_layout(width=1000)
    figure.write_image(const.PATH_IMG + 'cons_ghi_scatter.svg')


# Переменные для записи двух частей датасета
df_p1 = None
df_p2 = None

# Переменные для записи обучающей и тестирующей частей
train_data = None
test_data = None

x_train = None
y_train = None

x_test = None
y_test = None

# Переменная для записи модели
model = None

# Переменная для записи прогноза на сутки
tmp_df = None

#Разделение датасета на 2 части
def split_df():
    global df_p1
    global df_p2
    global tmp_df

    tmp_df = pd.read_csv('./res/solar_weather.csv')
    tmp_df = tmp_df.drop(['month', 'hour'], axis=1)
    tmp_df['Time'] = pd.to_datetime(tmp_df['Time'])

    df['month'] = df.index.month

    df_p1 = df[df['month'].isin(range(1, 12))]
    tmp_df.drop(tmp_df.index[96:196776], inplace=True)
    tmp_df = tmp_df.set_index('Time')
    df_p2 = tmp_df

    df_p2['month'] = tmp_df.index.month

    df_p2.to_csv(const.PATH_CSV + 'december_true.csv', index=False)


# Разделение датасета на тренировачные и тестирующие данные
def split_train_test():
    global df_p1
    global x_train, y_train, train_data, test_data
    global x_test, y_test

    df_p1 = df_p1.reset_index(drop=True)

    train_data, test_data = train_test_split(df_p1, test_size=0.2)

    x_train = train_data.drop('Energy delta[Wh]', axis=1).values.reshape(-1, 1, 14)
    x_test = test_data.drop('Energy delta[Wh]', axis=1).values.reshape(-1, 1, 14)
    y_train = train_data['Energy delta[Wh]'].values.reshape(-1, 1)
    y_test = test_data['Energy delta[Wh]'].values.reshape(-1, 1)


# Переменная для хранения прогноза модели
xgb_preds = None


# Создание XGBoost модели
def make_model():
    global model, xgb_preds
    global train_data, test_data

    features = const.COLUMNS
    target = "Energy delta[Wh]"

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.03,
        max_depth=20,
        subsample=0.9,
        colsample_bytree=0.3,
        n_estimators=1000,
        random_state=42
    )

    model.fit(train_data[features], train_data[target])
    xgb_preds = model.predict(tmp_df[features])


# Обучение модели
def train_model():
    model.fit(x_train, y_train, epochs=10, verbose=1)


predict = None


# Прогноз
def predict_december():
    global df_p2, model, x_test, predict, test_data

    df_p2 = df_p2.reset_index(drop=True)

    december_predicts = model.predict(tmp_df[const.COLUMNS])
    df_p2["Energy delta[Wh]"] = december_predicts.flatten()
    df_p2.to_csv(const.PATH_CSV + "december_predict.csv", index=False)


december_true = None
december_predict = None


# Запись результатов прогноза в файл
def write_results():
    global december_true, december_predict
    december_true = pd.read_csv(const.PATH_CSV + 'december_true.csv')
    december_predict = pd.read_csv(const.PATH_CSV + 'december_predict.csv')

    with open(const.PATH_TXT + 'december_true.txt', 'w',
              encoding='utf-8') as file:
        file.write(december_true.to_string())

    with open(const.PATH_TXT + 'december_predict.txt', 'w',
              encoding='utf-8') as file:
        file.write(december_predict.to_string())


# Рассчёт и запись метрик работы модели в файл
def calc_metrics():
    global december_true, december_predict
    global y_test, predict, tmp_df

    test = tmp_df['Energy delta[Wh]'].values.reshape(-1, 1)

    mapedf = np.mean(np.abs((december_true['Energy delta[Wh]'] - december_predict['Energy delta[Wh]']) / december_true['Energy delta[Wh]'])) * 100
    mape = np.mean(np.abs(test - xgb_preds) / np.maximum(test, 1)) * 100
    mae = mean_absolute_error(test, xgb_preds)
    mse = mean_squared_error(test, xgb_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(test, xgb_preds)

    with open(const.PATH_TXT + 'model_result.txt', 'w',
              encoding='utf-8') as file:
        file.write('Метрики работы модели:\n'
                   '----------------------------\n'
                   'Model Percentage Mean Absolute Error: ' + str(mape) + '\n'
                   'Mean Absolute Error: ' + str(mae) + '\n'
                   'Mean Squared Error: ' + str(mse) + '\n'
                   'Root Mean Squared Error: ' + str(rmse) + '\n'
                   'R^2: ' + str(r2) + '\n'
                   'Percentage Mean Absolute Error: ' + str(mapedf) + '\n'
                   '----------------------------\n')
