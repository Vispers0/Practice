import func


def main():
    # Разведочный анализ данных
    func.write_df()
    func.write_info()
    # func.make_column()
    func.write_desc()
    func.group_df()
    func.build_matrix()

    # Построение графиков
    func.build_general_plot()
    func.build_box_plot()
    func.build_general_bar()
    func.build_ghi_bar()
    func.build_consumption_box()
    func.build_ghi_box()
    func.build_cons_ghi_scatter()

    #Построение модели и прогнозирование
    # func.split_df()
    # func.split_train_test()
    # func.make_lstm()
    # func.train_model()
    # func.predict_december()
    # func.write_results()
    # func.calc_metrics()


if __name__ == '__main__':
    main()
