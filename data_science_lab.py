### Импорт модулей
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay
sns.set()
###


### Параметры по умолчанию
# Глобальные параметры
# pd.set_option('display.max_rows', None)     # Снятие ограничений по строкам  (вывод таблиц)
# pd.set_option('display.max_columns', None)  # Снятие ограничений по столбцам (вывод таблиц)
random_state = 69  # Параметр воспроизводимости случайностей

# 1 задание
csv_to_import = 'Shill Bidding Dataset.csv'  # Название файла для импорта

# 4 Задание
columns_to_remove = ['Record_ID', 'Auction_ID', 'Bidder_ID']  # Список ненужных атрибутов

# 5 Задание
shuffle_amount = 20  # Количество вариантов перемешиваний
shuffle_default = 11  # Перемешивание по умолчанию
columns_targets = ['Class']  # Целевые атрибуты

# 6 Задание
neighbors_amount = False  # Пользовательское количество соседей | False - чтобы оставить по умолчанию

# 8 Задание
leaf_range = range(20, 81, 2)  # Диапазон изменения параметра регуляризации
###



### (1) Импортировать данные из CSV-файла
def data_import_csv(csv_name):
    return pd.read_csv(csv_name)
###
data_imported = data_import_csv(csv_to_import)



### (2) Определить количество записей и полей и их типі
def data_count_rows_columns(dataframe):
    return dataframe.shape, dataframe.dtypes
###



### (4) Удалить ненужные атрибуты
def data_remove_waste(dataframe, remove_list=[]):
    return dataframe.drop(columns=remove_list, errors='ignore')
###
data_imported_cleared = data_remove_waste(data_imported, columns_to_remove) #Необязательно



### (5) Взболтать, а не смешивать
def data_shuffle_n_split(dataframe, targets, amount, index, random_state=random_state, console_output=False):
    dataframe_features = dataframe.drop(columns=targets, errors='ignore')
    dataframe_targets = dataframe[targets]
    splitter = ShuffleSplit(amount, random_state=random_state)
    iteration = 1

    for train_index, test_index in splitter.split(dataframe_features, dataframe_targets):
        if console_output:
            print(f'-----------Перемешивание {iteration}-----------')
            print("Обучающая:\n", dataframe_features.iloc[train_index], dataframe_targets.iloc[train_index], "\nКонтрольная:\n", dataframe_features.iloc[test_index], dataframe_targets.iloc[test_index])

        if iteration == index:
            train_features, test_features, train_targets, test_targets = dataframe_features.iloc[train_index], dataframe_features.iloc[test_index], dataframe_targets.iloc[train_index],  dataframe_targets.iloc[test_index]

        iteration += 1

    return train_features, test_features, train_targets, test_targets

def data_check_balance(targets):
    zero_percantage = (targets[targets.Class == 0].count() / targets.count())[0]
    one_percantage = 1 - zero_percantage
    print(f'Class 0: {round(zero_percantage * 100, 2)}%\nClass 1: {round(one_percantage * 100, 2)}%')
    if .4 < zero_percantage < .6:
        print(f'Выборка сбалансирована ✓')
    else:
        print(f'Выборка несбалансирована x')
###


### (6) Построить классификационную модель
def data_kneighbors_build(train_features, train_targets):
    classifier = KNeighborsClassifier()
    classifier = classifier.fit(train_features, train_targets)

    return classifier
###


### (7) Вычислить и сравнить классификационные метрики
def data_find_metrics(classifier, features, targets):
    prediction = classifier.predict(features)
    return precision_score(targets, prediction), recall_score(targets, prediction), accuracy_score(targets, prediction), f1_score(targets, prediction), roc_auc_score(targets, prediction)


def data_criterions_output(classifier, train_features, train_targets, test_features, test_targets):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    fig.suptitle('Эффективность работы классификатора')


    RocCurveDisplay.from_estimator(classifier, test_features, test_targets, ax=axes[0], name='Контрольая')
    RocCurveDisplay.from_estimator(classifier, train_features, train_targets, ax=axes[0], name='Обучающая')
    axes[0].set(title='ROC-curve', ylabel='True Positive Rate', xlabel='False Positive Rate', ylim=(-0.05, 1.05))

    PrecisionRecallDisplay.from_estimator(classifier, test_features, test_targets, ax=axes[1], name='Контрольная')
    PrecisionRecallDisplay.from_estimator(classifier, train_features, train_targets, ax=axes[1], name='Обучающая')
    axes[1].set(title='PR-curve', ylabel='Precision', xlabel='Recall', ylim=(-0.05, 1.05))


    ConfusionMatrixDisplay.from_estimator(classifier, train_features, train_targets, ax=axes[2], cmap="YlGnBu")
    axes[2].set(title='Confusion matrix (обучающая)', ylabel='Real data', xlabel='Predictions')
    axes[2].set_anchor('C')
    axes[2].grid(False)

    ConfusionMatrixDisplay.from_estimator(classifier, test_features, test_targets, ax=axes[3], cmap="YlGnBu")
    axes[3].set(title='Confusion matrix (контрольная)', ylabel='Real data', xlabel='Predictions')
    axes[3].set_anchor('C')
    axes[3].grid(False)


    plt.tight_layout()
    plt.show()


###


### (8) Определить влияение размера листьев
def data_check_impact(train_features, train_targets, parameter_range):
    param_grid = {
        'leaf_size': parameter_range
    }

    kdtree_grid_search = GridSearchCV(KNeighborsClassifier(algorithm='kd_tree'), param_grid, scoring='accuracy')
    kdtree_grid_search.fit(train_features, train_targets)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes = axes.flatten()

    axes[0].plot(kdtree_grid_search.cv_results_['param_leaf_size'].data, kdtree_grid_search.cv_results_['mean_test_score'])
    axes[0].set(title='Влияние на точность', ylabel='Accuracy', xlabel='Размер листа')

    axes[1].plot(kdtree_grid_search.cv_results_['param_leaf_size'].data, kdtree_grid_search.cv_results_['std_fit_time'])
    axes[1].set(title='Влияние на скорость обучения', ylabel='Скорость обучения', xlabel='Размер листа')

    plt.tight_layout()
    plt.show()
###













### Вспомогательные функции
def choose_correct_int(title = ''):
    while True:
        try:
            return int(input(title))
        except:
            print('Введите корректное число!')
###



### Меню взаимодействия с пользователем
def main_menu(shuffle_index=shuffle_default):
    train_features, test_features, train_targets, test_targets = data_shuffle_n_split(data_imported_cleared, columns_targets, shuffle_amount, shuffle_index)
    classifier_regression = data_kneighbors_build(train_features, train_targets)

    print('\n' * 50)
    print('Даниленко Кирилл КМ-92')
    print('Лабораторная №4')
    print(f'Перемешивание №{shuffle_index}')

    print('\nСписок действий:')
    menu_items = [
        '1. Показать импортированные данные',
        '2. Определить количество записей и полей',
        '3. Вывести атрибуты набора данных',
        '4. Удалить ненужные для анализа атрибуты',
        '5. Выбрать перемешивание (1 - 20)',
        '6. Построить классификационную модель',
        '7. Вычислить и сравнить классификационные метрики',
        '8. Определить влияение размера листьев',
    ]

    for item in menu_items:
        print(item)

    user_choice = choose_correct_int('\nВыберите один из пунктов: ')
    if user_choice == 1:
        print('\n' * 50)
        print(f'\nИмпортированные данные: \n{data_imported}\n')
    elif user_choice == 2:
        print('\n' * 50)
        amounts, types = data_count_rows_columns(data_imported)
        print(f'Количество записей: {amounts[0]}\nКоличество полей: {amounts[1]}\n\nТипы полей:\n{types}\n')
    elif user_choice == 3:
        print('\n' * 50)
        print(f'Атрибуты набора данных: \n{data_imported.drop(columns=columns_targets, errors="ignore")}')
    elif user_choice == 4:
        print('\n' * 50)
        print(f'Очищенная выборка (удалены {", ".join(columns_to_remove)}):\n{data_imported_cleared}')
    elif user_choice == 5:
        print('\n' * 50)
        user_index = choose_correct_int(f'\nТекущее перемешивание: №{shuffle_index}\nВыберите новое перешивание (1 - {shuffle_amount}): ')
        print('\n' * 50)

        print(f'Текущее перемешивание (№{shuffle_default}): \n{train_features}')
        data_check_balance(train_targets)

        if 1 <= user_index <= shuffle_amount:
            shuffle_index = user_index
            train_features, test_features, train_targets, test_targets = data_shuffle_n_split(data_imported_cleared, columns_targets, shuffle_amount, index=shuffle_index)

        print(f'\n\nНовое перемешивание (№{shuffle_index}): \n{train_features}')
        data_check_balance(train_targets)
    elif user_choice == 6:
        print('\n' * 50)
        print(f'Она построилась, сомнений нет:\n{classifier_regression}')
    elif user_choice == 7:
        print('\n' * 50)

        print('<<Обучающая выборка>>\nPrecision: {}\nRecall: {}\nAccuracy: {}\nF1 Score: {}\nROC-AUC: {}'.format(*data_find_metrics(classifier_regression, train_features, train_targets)))
        print('\n\n<<Тестовая выборка>>\nPrecision: {}\nRecall: {}\nAccuracy: {}\nF1 Score: {}\nROC-AUC: {}'.format(*data_find_metrics(classifier_regression, test_features, test_targets)))

        data_criterions_output(classifier_regression, train_features, train_targets, test_features, test_targets)
    elif user_choice == 8:
        print('\n' * 50)
        data_check_impact(train_features, train_targets, leaf_range)
    else:
        main_menu(shuffle_index)


    print('\nПрограмма завершила работу.\nВернуться в главное меню? (Y/N)')
    if input().lower() == 'y':
        main_menu(shuffle_index)
###



### Старт программы
main_menu()
###
