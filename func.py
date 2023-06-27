import pandas as pd
import numpy as np
import scipy.stats as stats
import datetime as dt

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

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