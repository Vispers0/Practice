import pandas as pd
import numpy as np
import datetime as dt

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

import io

import const

df_grouped = None

df = pd.read_csv("./res/solar_weather.csv")


# Запись датасета в файл
def write_df():
    with open(const.PATH + const.DF, 'w',
              encoding='utf-8') as file:
        file.write(df.to_string())


# Запись информации о датасете в файл
def write_info():
    buf = io.StringIO()
    df.info(buf=buf)
    with open(const.PATH + const.INFO, 'w',
              encoding='utf-8') as file:
        file.write(buf.getvalue())
        file.write("amount of rows and columns: " + str(df.shape))
        file.write("\namount of null cells:\n" + str(df.isnull().sum()))
        file.write("\namount of duplicated values: " + str(df.duplicated().sum()))


# Создание нового столбца для группировки
def make_column():
    df['Time'] = pd.to_datetime(df['Time'])
    df['year'] = df['Time'].dt.year


# Запись описательной статистики в файл
def write_desc():
    with open(const.PATH + const.DESC, 'w',
              encoding='utf-8') as file:
        file.write(df.describe().round(3).to_string())


# Группировка датасета по году, месяцу и часу
def group_df():
    global df_grouped
    df_grouped = df.groupby(['year', 'month', 'hour']).mean(numeric_only=True).round(3)

    df_grouped['isSun'] = df_grouped['isSun'].astype(float).round(0)
    df_grouped['weather_type'] = df_grouped['weather_type'].astype(float).round(0)

    with open(const.PATH + const.GROUP, 'w',
              encoding='utf-8') as file:
        file.write(df_grouped.to_string())


# Построение матрицы корреляции
def build_matrix():
    df_corr = df
    df_corr = df_corr.drop('Time', axis=1)
    df_corr = df_corr.corr()

    x = list(df_corr.columns)
    y = list(df_corr.index)
    z = np.array(df_corr)

    figure = ff.create_annotated_heatmap(x=x,
                                         y=y,
                                         z=z,
                                         annotation_text=np.around(z, 2))

    figure.layout.width = 1000
    figure.layout.height = 1000

    figure.write_image("./output/corr_matrix.svg")
