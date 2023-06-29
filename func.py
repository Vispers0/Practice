import pandas as pd
import numpy as np
import datetime as dt

import plotly.express as px
import plotly.figure_factory as ff

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

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
    figure = px.scatter(df_grouped,
                        x='GHI',
                        y='Energy delta[Wh]',
                        title='Зависимость потребления электроэнергии от глобального горизонтального излучения')

    figure.update_layout(width=1000)
    figure.write_image(const.PATH_IMG + 'cons_ghi_scatter.svg')


df_p1 = None
df_p2 = None

train_data = None
x_train = None
y_train = None

model = None


#Разделение датасета на 2 части
def split_df():
    global df_p1
    global df_p2

    df['month'] = df.index.month

    df_p1 = df[df['month'].isin(range(1, 12))]
    df_p2 = df[df['month'] == 12]


# Разделение датасета на тренировачные и тестирующие данные
def split_train_test():
    global df_p1
    global x_train, y_train, train_data

    df_p1 = df_p1.reset_index(drop=True)

    train_data, test_data = train_test_split(df_p1, test_size=0.2, random_state=42)

    x_train = train_data.drop('Energy delta[Wh]', axis=1).values.reshape(-1, 1, 14)
    x_test = test_data.drop('Energy delta[Wh]', axis=1).values.reshape(-1, 1, 14)
    y_train = train_data['Energy delta[Wh]'].values.reshape(-1, 1)
    y_test = test_data['Energy delta[Wh]'].values.reshape(-1, 1)

    print("x shape" + str(x_train.shape))
    print("y shape" + str(y_train.shape))


# Создание LSTM модели
def make_lstm():
    global model
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, 14)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    with open(const.PATH_TXT + 'lstm_model.txt', 'w',
              encoding='utf-8') as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))


# Обучение модели
def train_model():
    model.fit(x_train, y_train, epochs=20, verbose=0)
