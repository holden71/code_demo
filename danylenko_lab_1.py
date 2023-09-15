### Импорт модулей
from numpy import mean, std, median, square
from scipy.stats import f, skew, mode, kurtosis, shapiro, normaltest
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import pandas as pd
###




### Глобальные параметры
# Глобальные параметры
#pd.set_option('display.max_rows', None)     # Снятие ограничений по строкам  (вывод таблиц)
#pd.set_option('display.max_columns', None)  # Снятие ограничений по столбцам (вывод таблиц)
###



### Импорт данных из файла
data_imported = pd.read_csv('A3.txt')

data_imported.columns = [f'A{i + 1}' for i in range(len(data_imported.columns))]
###



### (1) Построить графики всех параметров
def data_columns_graph(dataframe):
    figure, axis = plt.subplots(ncols=2, nrows=6, figsize=(15, 15))
    axis = axis.flatten()

    column_index = 0

    for column_name, column_data in dataframe.iteritems():
        axis[column_index].plot(column_data, color='orange')
        axis[column_index].set(title=f'Параметр {column_name}')

        column_index += 1

    plt.tight_layout()
    plt.show() # Вывод графиков
###



### (2) Определить статистические параметры + Гистограмма
def data_parametrs_print(dataframe):
    figure, axis = plt.subplots(ncols=2, nrows=6, figsize=(15, 15))
    axis = axis.flatten()

    result_table = PrettyTable()

    result_table.add_column('Параметры', [
        'Среднее значение',
        'Дисперсия',
        'Мода',
        'Медиана',
        'Коэффициент ассиметрии',
        'Коэффициент эксцесса',
        'Тест Шапиро',
        'Тест на норм. распред'
    ])

    column_index = 0
    for column_name, column_data in dataframe.iteritems():
        result_table.add_column(column_name, [
            round(mean(column_data), 2),
            round(std(column_data), 2),
            round(mode(column_data)[0][0], 2),
            round(median(column_data), 2),
            round(skew(column_data), 2),
            round(kurtosis(column_data), 2),
            round(shapiro(column_data)[0], 2),
            'Да' if normaltest(column_data)[1] > 1e-3 else 'Нет'
        ])

        column_data.hist(ax=axis[column_index])
        column_data.plot.kde(ax=axis[column_index], secondary_y=True)
        axis[column_index].set(title=f'Параметр {column_name}', ylabel='Плотность')

        column_index += 1

    print(result_table) # Вывод статистических параметров
    plt.tight_layout()
    plt.show() # Вывод гистограмм
###



### (3) Провести однофакторный анализ
def data_single_factor_analysis(dataframe):
    result_table = PrettyTable()

    si_squared_list = []
    N_value = len(dataframe)
    K_value = len(dataframe.columns)
    sum_of_sum_squared = 0
    sum_of_squared_sum = 0
    sum_of_sum = 0

    for column_name, column_data in dataframe.iteritems():
        sum_of_squared = pow(column_data, 2).sum()
        square_of_sum = pow(column_data.sum(), 2)

        si_squared = (sum_of_squared - square_of_sum/N_value)/(N_value-1)
        si_squared_list.append(round(si_squared, 3))

        sum_of_sum_squared += sum_of_squared
        sum_of_squared_sum += square_of_sum
        sum_of_sum += sum(column_data)

    result_table.field_names = dataframe.columns
    result_table.add_row(si_squared_list)
    print(result_table)  # Вывод параметров факторов

    si_squared_max = max(si_squared_list)
    si_squared_sum = sum(si_squared_list)
    si_squared_g = si_squared_max / si_squared_sum
    si_squared_g_alfa = 0.5

    print(f'\nМаксимум Si^2: {si_squared_max}')
    print(f'Сумма Si^2: {si_squared_sum}')
    print(f'Оценка параметра g: {si_squared_g}')
    print(f'Параметр g_alpha: {si_squared_g_alfa}')

    if si_squared_g < si_squared_g_alfa:
        print('<<Гипотеза о равенстве дисперсий принимается>>')
    else:
        print('<<Гипотеза о равенстве дисперсий отклоняется>>')
        return

    so_squared = (sum_of_sum_squared - sum_of_squared_sum/N_value) / (K_value*(N_value - 1))

    sa_squared = (N_value / (K_value - 1)) * (square(dataframe.mean() - dataframe.mean().mean()).sum())

    print(f'\nОценка рассеивания вне фактора: {so_squared}')
    print(f'Оценка рассеивания из-за фактора: {sa_squared}')

    if sa_squared / so_squared > f.ppf(0.95, (K_value - 1), K_value * (N_value - 1)):
        print('<<Влияние фактора является значительным>>')
    else:
        print('<<Влияние фактора является незначительным>>')
###



### (4) Провести двухфакторный анализ
def data_double_factor_analysis(dataframe):
    N_value = len(dataframe)
    K_value = len(dataframe.columns)
    M_value = 5
    alpha = 0.95
    a_significant = False
    b_significant = False

    dataframe_splitted = dataframe.groupby(dataframe.index // 1000).agg(list)
    dataframe_splitted_mean = dataframe_splitted.applymap(mean)
    dataframe_splitted_sum_square = dataframe_splitted.applymap(square).applymap(sum)

    q1_value = square(dataframe_splitted_mean.values).sum()
    q2_value = square(dataframe_splitted_mean.sum(axis=0)).sum() / M_value
    q3_value = square(dataframe_splitted_mean.sum(axis=1)).sum() / K_value
    q4_value = square(dataframe_splitted_mean.sum(axis=1).sum()) / (M_value * K_value)

    s_squared_0 = (q1_value + q4_value - q2_value - q3_value) / ((K_value - 1) * (M_value - 1))
    s_squared_a = (q2_value - q4_value) / (K_value - 1)
    s_squared_b = (q3_value - q4_value) / (M_value - 1)



    print('Значение S2_0:', s_squared_0)
    print('Значение S2_A:', s_squared_a)
    print('Значение S2_B:', s_squared_b)



    if s_squared_a / s_squared_0 > f.ppf(alpha, (K_value - 1), K_value * (N_value - 1)):
        print('<<Влияние фактора А является значительным>>')
        a_significant = True
    else:
        print('<<Влияние фактора А является незначительным>>')

    if s_squared_a / s_squared_b > f.ppf(alpha, (K_value - 1), K_value * (N_value - 1)):
        print('<<Влияние фактора B является значительным>>')
        b_significant = True
    else:
        print('<<Влияние фактора B является незначительным>>')



    if a_significant and b_significant:
        q5_value = dataframe_splitted_sum_square.to_numpy().sum()
        s_squared_ab = (q5_value - N_value * q1_value) / (M_value * K_value * (N_value - 1))

        if N_value * s_squared_0 / s_squared_ab > f.ppf(alpha, (K_value - 1) * (M_value - 1), M_value * K_value * (N_value - 1)):
            print('<<Влияние фактора АB является значительным>>')
    else:
        print('<<Влияние фактора АB является незначительным>>')
###

















### Вспомогательные функции
def choose_correct_int(title = ''):
    if title:
        print(title)

    while True:
        try:
            return int(input())
        except:
            print('Введите корректное число!')
###



### Меню взаимодействия с пользователем
def main_menu():
    print('\n' * 50)
    print('Даниленко Кирилл КМ-92')
    print('Лабораторная №1')

    print('\nСписок заданий:')
    menu_items = [
        '1. Построить графики всех параметров',
        '2. Определить статистические параметры и построить гистограмму',
        '3. Провести однофакторный анализ',
        '4. Провести двухфакторный анализ',
    ]

    for item in menu_items:
        print(item)

    user_choice = choose_correct_int('\nВыберите один из пунктов:')
    if user_choice == 1:
        print('\n' * 50)
        data_columns_graph(data_imported)
    elif user_choice == 2:
        print('\n' * 50)
        data_parametrs_print(data_imported)
    elif user_choice == 3:
        print('\n' * 50)
        data_single_factor_analysis(data_imported)
    elif user_choice == 4:
        print('\n' * 50)
        data_double_factor_analysis(data_imported)
    else:
        main_menu()



    print('\nПрограмма завершила работу.\nВернуться в главное меню? (Y/N)')
    if input().lower() == 'y':
        main_menu()
###



### Старт программы
main_menu()
###