import warnings

import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def preprocessing(encode_cat=False):
    """
    Главная функция для препроцесса данных
    Returns:
        (train, test, num_columns, cat_columns, target),
            where train and test are preprocessed dataframes,
            num_columns is list of all numerical columns (without target),
            cat_columns is list of all categorical columns (without target),
            target is target column
    """

    # Чтение файлов
    df_chronom_test = pd.read_csv('data/chronom_test.csv')
    df_chronom_train = pd.read_csv('data/chronom_train.csv')

    df_chugun_test = pd.read_csv('data/chugun_test.csv')
    df_chugun_train = pd.read_csv('data/chugun_train.csv')

    df_gas_test = pd.read_csv('data/gas_test.csv')
    df_gas_train = pd.read_csv('data/gas_train.csv')
    df_gas_test['Time'] = pd.to_datetime(df_gas_test['Time'])
    df_gas_train['Time'] = pd.to_datetime(df_gas_train['Time'])

    df_lom_test = pd.read_csv('data/lom_test.csv')
    df_lom_train = pd.read_csv('data/lom_train.csv')

    df_plavki_test = pd.read_csv('data/plavki_test.csv')
    df_plavki_train = pd.read_csv('data/plavki_train.csv')

    df_produv_test = pd.read_csv('data/produv_test.csv')
    df_produv_train = pd.read_csv('data/produv_train.csv')
    df_produv_train['SEC'] = pd.to_datetime(df_produv_train['SEC'])
    df_produv_test['SEC'] = pd.to_datetime(df_produv_test['SEC'])

    df_sip_test = pd.read_csv('data/sip_test.csv')
    df_sip_train = pd.read_csv('data/sip_train.csv')

    target = pd.read_csv('data/target_train.csv')
    sample_submission = pd.read_csv('data/sample_submission.csv')
    ##############################################
    print('Чтение закончилось')

    # Функция времен для chronom df
    # Kartushov Danil
    def get_times(train, test, get_sec=True, fillnao2=True, clear_df=False):
        """
        Return filtred times in Chrono dataset : train, test in order
        :train: train dataset
        :test: test dataset
        :get_sec: additional column difference date in seconds
        """

        train['VR_NACH'] = pd.to_datetime(train['VR_NACH'])
        test['VR_NACH'] = pd.to_datetime(test['VR_NACH'])
        train['VR_KON'] = pd.to_datetime(train['VR_KON'])
        test['VR_KON'] = pd.to_datetime(test['VR_KON'])

        train['time_diff'] = train['VR_KON'] - train['VR_NACH']
        test['time_diff'] = test['VR_KON'] - test['VR_NACH']
        # время в секундах
        if get_sec == True:
            train['time_diff_sec'] = train['time_diff'].dt.total_seconds()
            test['time_diff_sec'] = test['time_diff'].dt.total_seconds()
        # Зануление O2 (в это время не подается кислород)
        if fillnao2 == True:
            train['O2'].fillna(0)
            test['O2'].fillna(0)
        # Чистка выбросов
        if clear_df == True:
            train.drop(index=[17247, 9112, 14698])

        return train, test

    df_chronom_train, df_chronom_test = get_times(df_chronom_train, df_chronom_test, get_sec=True)

    # Отбор NOP, которые есть только в test&test
    # df_chronom_train = df_chronom_train[df_chronom_train['NOP'].isin(df_chronom_test['NOP'].unique())]

    #################
    # Мерж трейн датасета
    train = df_chugun_train.copy(deep=True)
    # Chugun
    num_columns = ['VES', 'T', 'SI', 'MN', 'S', 'P', 'CR', 'NI', 'CU', 'V', 'TI']

    df_plavki_train['plavka_NMZ'] = df_plavki_train['plavka_NMZ'].apply(lambda x: x.replace(' ', '').lower())
    df_plavki_test['plavka_NMZ'] = df_plavki_test['plavka_NMZ'].apply(lambda x: x.replace(' ', '').lower())

    train = train.merge(df_plavki_train, on='NPLV')

    # Plavki

    num_columns += ['plavka_STFUT', 'plavka_ST_FURM', 'plavka_ST_GOL']

    cat_columns = ['plavka_NMZ', 'plavka_NAPR_ZAD', 'plavka_TIPE_FUR', 'plavka_TIPE_GOL']

    train = train.merge(target, on='NPLV')
    # Target
    target_columns = ['C', 'TST']

    # Мерж тест датасета
    test = df_chugun_test.copy(deep=True)
    test = test.merge(df_plavki_test, on='NPLV')

    # Олег - новые временные фичи - длина плавки в секундах и конец плавки - дата замера чугуна
    train['plavka_VR_KON'] = pd.to_datetime(train['plavka_VR_KON'])
    train['plavka_VR_NACH'] = pd.to_datetime(train['plavka_VR_NACH'])
    train['DATA_ZAMERA'] = pd.to_datetime(train['DATA_ZAMERA'])
    train['plavka_time'] = (train['plavka_VR_KON'] - train['plavka_VR_NACH']).dt.total_seconds()
    train['timediff_chugun'] = (train['plavka_VR_KON'] - train['DATA_ZAMERA']).dt.total_seconds() / train['plavka_time']

    test['plavka_VR_KON'] = pd.to_datetime(test['plavka_VR_KON'])
    test['plavka_VR_NACH'] = pd.to_datetime(test['plavka_VR_NACH'])
    test['DATA_ZAMERA'] = pd.to_datetime(test['DATA_ZAMERA'])
    test['plavka_time'] = (test['plavka_VR_KON'] - test['plavka_VR_NACH']).dt.total_seconds()
    test['timediff_chugun'] = (test['plavka_VR_KON'] - test['DATA_ZAMERA']).dt.total_seconds()

    num_columns += ['plavka_time', 'timediff_chugun']

    ######################

    # Даниил: Температура работающей фурмы
    train = pd.merge(train, chosen_furma_tempa(df_gas_train), how='left', left_on='NPLV', right_on='NPLV')
    test = pd.merge(test, chosen_furma_tempa(df_gas_test), how='left', left_on='NPLV', right_on='NPLV')
    furma_nan_df(train)
    furma_nan_df(test)
    num_columns += ['furma_chosen_mean', 'furma_chosen_max']

    # Олег: веса ломов по видам
    train_lom = lom_weights_df(df_lom_train)
    test_lom = lom_weights_df(df_lom_test)
    train = pd.merge(train, train_lom, how='left', left_on='NPLV', right_on='NPLV')
    test = pd.merge(test, test_lom, how='left', left_on='NPLV', right_on='NPLV')

    num_columns += ['VES0', 'VES1', 'VES2', 'VES3', 'VES4', 'VES5', 'VES6', 'VES7']

    # Олег: веса сыпучих по видам
    train_sip = sip_weights_df(df_sip_train)
    test_sip = sip_weights_df(df_sip_test)
    train = pd.merge(train, train_sip, how='left', left_on='NPLV', right_on='NPLV')
    test = pd.merge(test, test_sip, how='left', left_on='NPLV', right_on='NPLV')

    num_columns += ['VSSYP104', 'VSSYP119', 'VSSYP171', 'VSSYP346', 'VSSYP397', 'VSSYP408', 'VSSYP442']

    # Олег/Даниил: фичи хронометража
    # train
    o2_pivot_train = df_chronom_train.pivot_table(index='NPLV', values=['O2'], aggfunc='sum')
    o2_timediff_pivot_train = df_chronom_train[df_chronom_train['NOP'] == 'Продувка'][
        ['NPLV', 'VR_NACH', 'VR_KON', 'time_diff', 'time_diff_sec']].set_index('NPLV')
    o2_final_pivot_train = pd.merge(o2_pivot_train, o2_timediff_pivot_train, left_index=True, right_index=True)
    # test
    o2_pivot_test = df_chronom_test.pivot_table(index='NPLV', values=['O2'], aggfunc='sum')
    o2_timediff_pivot_test = df_chronom_test[df_chronom_test['NOP'] == 'Продувка'][
        ['NPLV', 'VR_NACH', 'VR_KON', 'time_diff', 'time_diff_sec']].set_index('NPLV')
    o2_final_pivot_test = pd.merge(o2_pivot_test, o2_timediff_pivot_test, left_index=True, right_index=True)

    print(o2_timediff_pivot_train.index[0:10])
    print(o2_timediff_pivot_train.index)

    df_produv_train = pd.read_csv('data/produv_train.csv')
    df_produv_test = pd.read_csv('data/produv_test.csv')
    df_produv_train['SEC'] = pd.to_datetime(df_produv_train['SEC'])
    df_produv_test['SEC'] = pd.to_datetime(df_produv_test['SEC'])

    # Олег/Даниил: аггрегация табличек 
    df_produv_filtered_train = get_filtered_produv(df_produv_train, o2_final_pivot_train)
    df_produv_filtered_test = get_filtered_produv(df_produv_test, o2_final_pivot_test)

    df_produv_aggregated_train = aggregate_produv(df_produv_filtered_train)
    df_produv_aggregated_test = aggregate_produv(df_produv_filtered_test)

    train = pd.merge(train, df_produv_aggregated_train, how='left', left_on='NPLV', right_on='NPLV')
    test = pd.merge(test, df_produv_aggregated_test, how='left', left_on='NPLV', right_on='NPLV')

    # Газ пока фурма (давление)
    df_gaspressure_filtered_train = get_filtered_gas_for_active_furm(df_gas_train, o2_final_pivot_train)
    df_gaspressure_filtered_test = get_filtered_gas_for_active_furm(df_gas_test, o2_final_pivot_test)

    df_gaspressure_aggregated_train = aggregate_pressure(df_gaspressure_filtered_train)
    df_gaspressure_aggregated_test = aggregate_pressure(df_gaspressure_filtered_test)

    train = pd.merge(train, df_gaspressure_aggregated_train, how='left', left_on='NPLV', right_on='NPLV')
    test = pd.merge(test, df_gaspressure_aggregated_test, how='left', left_on='NPLV', right_on='NPLV')

    num_columns += ['produv_mean_RAS', 'produv_mean_POL', 'produv_sum_RAS', 'produv_sum_POL', 'produv_min_RAS',
                    'produv_min_POL',
                    'produv_max_RAS', 'produv_max_POL', 'produv_std_RAS', 'produv_std_POL',
                    'produv_amp_RAS', 'produv_amp_POL', 'pressure_mean_O2_pressure',
                    'pressure_sum_O2_pressure', 'pressure_min_O2_pressure',
                    'pressure_max_O2_pressure', 'pressure_std_O2_pressure', 'pressure_amp_O2_pressure']

    # Все остальные газы

    o2_final_pivot_train = pd.merge(o2_pivot_train, o2_timediff_pivot_train, left_index=True,
                                    right_index=True).reset_index()
    o2_final_pivot_test = pd.merge(o2_pivot_test, o2_timediff_pivot_test, left_index=True,
                                   right_index=True).reset_index()

    o2_final_pivot_test.columns = ['NPLV', 'chronom_O2', 'chronom_produv_NACH', 'chronom_produv_KON',
                                   'chronom_produv_td', 'chronom_produv_timediff']
    o2_final_pivot_train.columns = ['NPLV', 'chronom_O2', 'chronom_produv_NACH', 'chronom_produv_KON',
                                    'chronom_produv_td', 'chronom_produv_timediff']

    train = pd.merge(train, o2_final_pivot_train, how='left', left_on='NPLV', right_on='NPLV')
    test = pd.merge(test, o2_final_pivot_test, how='left', left_on='NPLV', right_on='NPLV')

    num_columns += ['chronom_O2', 'chronom_produv_timediff']

    ############################## 
    train = pd.merge(train, return_gas_volumes(df_gas=df_gas_train), how='left', left_on='NPLV', right_on='NPLV')
    test = pd.merge(test, return_gas_volumes(df_gas=df_gas_test), how='left', left_on='NPLV', right_on='NPLV')
    num_columns += ['O2_volume', 'N2_volume', 'H2_volume', 'CO2_volume', 'CO_volume', 'AR_volume', 'out_C']

    # Артем: тепло отходящих газов - объем на температуру
    train = pd.merge(train, return_gas_heat(df_gas=df_gas_train), how='left', left_on='NPLV', right_on='NPLV')
    test = pd.merge(test, return_gas_heat(df_gas=df_gas_test), how='left', left_on='NPLV', right_on='NPLV')
    num_columns += ['gas_heat']

    # Артем: время между плавками 
    train = pd.merge(train, time_between_plavki(df_chronom=df_chronom_train), how='left', left_on='NPLV',
                     right_on='NPLV')
    test = pd.merge(test, time_between_plavki(df_chronom=df_chronom_test), how='left', left_on='NPLV', right_on='NPLV')
    num_columns += ['time_betw_plavki']

    # Артем: добавление целевых значений по углероду
    df_marki_stali = pd.read_csv('data/marki_stali.csv')
    train = pd.merge(train, df_marki_stali, how='left', left_on='plavka_NMZ', right_on='plavka_NMZ')
    test = pd.merge(test, df_marki_stali, how='left', left_on='plavka_NMZ', right_on='plavka_NMZ')
    num_columns += ['C_min', 'C_max']

    print(1)
    # Дроп лишних колонок
    print(train.columns)
    bad_columns = ['DATA_ZAMERA', 'plavka_VR_NACH', 'plavka_VR_KON', 'index_x', 'index_y', 'VES8', 'index',
                   'chronom_produv_NACH', 'chronom_produv_KON', 'chronom_produv_td']
    train.drop(bad_columns, axis=1, inplace=True, errors='ignore')
    test.drop(bad_columns, axis=1, inplace=True, errors='ignore')

    # ЧИСТКА
    clean_data(train)
    clean_data(test)

    ########### Дроп мусорной плавки
    NPLV_to_drop = [511135, 511156, 512299, 512322]
    # 511135 багнутые плавки 75 штук,  511156, 512299 nan в C, 512322 - багнутый хвост на 10 дней продувка

    train.drop(train[train.isin(NPLV_to_drop)['NPLV'] == True].index, inplace=True)
    ####################################

    # Энкоды + проверки
    if encode_cat:
        # Label encoding for categorical columns
        for column in cat_columns:
            le = LabelEncoder()
            le.fit(pd.concat([train[column], test[column]]))
            train[column] = le.transform(train[column]).astype(int)
            test[column] = le.transform(test[column]).astype(int)

    for column in num_columns:
        train[column] = train[column].astype(float)
        test[column] = test[column].astype(float)

    # Type check for everything
    if encode_cat:
        for column in cat_columns:
            assert train[column].dtype == 'int', (column, train[column].dtype)
            assert test[column].dtype == 'int', (column, test[column].dtype)

    for column in num_columns:
        assert train[column].dtype == 'float', (column, train[column].dtype)
        assert test[column].dtype == 'float', (column, test[column].dtype)
        pass

    return train, test, num_columns, cat_columns, target_columns  # Возвращать только так!


def lom_weights_df(df):
    # Олег веса по виду лома
    df.drop('VDL', axis=1, inplace=True)

    df = df.groupby(['NPLV', 'NML']).sum().unstack()
    df.columns = [val1 + val2 for (val1, val2) in df.columns]
    df['index'] = range(len(df))
    df.reset_index().set_index('index')
    # Drop rare columns
    df.drop(['VESСК  ', 'VESНБ  '], errors='ignore', axis=1, inplace=True)  # колонки там 5 экземпляров и 1
    df.columns = [f"VES{i}" for i in range(len(df.columns))]
    # All nans are zero
    df.fillna(0, inplace=True)
    return df


def sip_weights_df(df):
    # Олег веса по виду сыпучих
    df.drop(['NMSYP', 'DAT_OTD'], axis=1, inplace=True)

    df = df.groupby(['NPLV', 'VDSYP']).sum().unstack()
    df.columns = [val1 + str(val2) for (val1, val2) in df.columns]
    df['index'] = range(len(df))
    df.reset_index().set_index('index')
    # Drop rare columns
    df.drop(['VSSYP344', 'VSSYP11'], errors='ignore', axis=1,
            inplace=True)  # Обоих нету в трейне, 11 очень мало в принципе
    # All nans are zero
    df.fillna(0, inplace=True)
    return df


def get_filtered_produv(df_produv, o2_final_pivot):
    df_produv_filtered = pd.DataFrame()
    for nplv_now in df_produv['NPLV'].unique():
        VR_NACH_now = o2_final_pivot.loc[nplv_now, 'VR_NACH']
        VR_KON_now = o2_final_pivot.loc[nplv_now, 'VR_KON']
        df_produv_now = df_produv[
            (df_produv['NPLV'] == nplv_now) &
            (df_produv['SEC'] >= VR_NACH_now) &
            (df_produv['SEC'] <= VR_KON_now)]
        df_produv_filtered = df_produv_filtered.append(df_produv_now)
    return df_produv_filtered


def aggregate_produv(df):
    # df это отфильтрованная таблица produv
    mean_df = df.groupby('NPLV').mean().add_prefix('produv_mean_')
    sum_df = df.groupby('NPLV').sum().add_prefix('produv_sum_')
    min_df = df.groupby('NPLV').min().drop('SEC', axis=1)
    max_df = df.groupby('NPLV').max().drop('SEC', axis=1)
    std_df = df.groupby('NPLV').std().add_prefix('produv_std_')
    amp_df = (max_df - min_df)
    amp_df = amp_df.add_prefix('produv_amp_')
    min_df = min_df.add_prefix('produv_min_')
    max_df = max_df.add_prefix('produv_max_')

    main_df = pd.concat([mean_df, sum_df, min_df, max_df, std_df, amp_df], axis=1).reset_index()
    return main_df


def aggregate_pressure(df):
    # df это отфильтрованная таблица gas!!!
    df = df[['NPLV', 'O2_pressure']]
    mean_df = df.groupby('NPLV').mean().add_prefix('pressure_mean_')
    sum_df = df.groupby('NPLV').sum().add_prefix('pressure_sum_')
    min_df = df.groupby('NPLV').min()
    max_df = df.groupby('NPLV').max()
    std_df = df.groupby('NPLV').std().add_prefix('pressure_std_')
    amp_df = (max_df - min_df)
    amp_df = amp_df.add_prefix('pressure_amp_')
    min_df = min_df.add_prefix('pressure_min_')
    max_df = max_df.add_prefix('pressure_max_')

    main_df = pd.concat([mean_df, sum_df, min_df, max_df, std_df, amp_df], axis=1).reset_index()

    return main_df


def aggregate_gasses(df):
    # df это отфильтрованная таблица gas!!!
    df = df[['NPLV', 'V', 'T', 'O2', 'N2', 'H2', 'CO2', 'CO', 'AR']]
    df['O2'] = df['O2'] * df['V']
    df['N2'] = df['N2'] * df['V']
    df['H2'] = df['H2'] * df['V']
    df['CO2'] = df['CO2'] * df['V']
    df['CO'] = df['CO'] * df['V']
    df['AR'] = df['AR'] * df['V']

    df['TV'] = df['T'] * df['V']

    df = df[['NPLV', 'T', 'O2', 'N2', 'H2', 'CO2', 'CO', 'AR', 'TV']]

    sum_df = df.groupby('NPLV').sum().add_prefix('gas_integral_').reset_index()

    return sum_df


def chugun_nan_df(df):
    # Олег нулевые составы чугуна в нан
    df.loc[df['SI'] == 0, 'MN'] = np.NaN
    df.loc[df['SI'] == 0, 'S'] = np.NaN
    df.loc[df['SI'] == 0, 'P'] = np.NaN
    df.loc[df['SI'] == 0, 'CR'] = np.NaN
    df.loc[df['SI'] == 0, 'NI'] = np.NaN
    df.loc[df['SI'] == 0, 'CU'] = np.NaN
    df.loc[df['SI'] == 0, 'V'] = np.NaN
    df.loc[df['SI'] == 0, 'TI'] = np.NaN
    df.loc[df['SI'] == 0, 'SI'] = np.NaN


def furma_nan_df(df):
    # Даня - фурмы
    df.loc[df['furma_chosen_mean'] == 0, 'furma_chosen_mean'] = np.NaN
    df.loc[df['furma_chosen_max'] == 0, 'furma_chosen_max'] = np.NaN


def clean_data(df):
    # Олег - очистка данных
    chugun_nan_df(df)


def chosen_furma_tempa(df_gas):
    """
    Features of furma tempreture
    """
    grouped_t_furma_df_gas = df_gas.groupby('NPLV').agg(
        mean_t_furma1=('T фурмы 1', 'mean'),
        mean_t_furma2=('T фурмы 2', 'mean'),
        max_t_furma1=('T фурмы 1', 'max'),
        max_t_furma2=('T фурмы 2', 'max'))

    grouped_t_furma_df_gas['furma_chosen_mean'] = grouped_t_furma_df_gas.apply(
        lambda x: max(x['mean_t_furma1'], x['mean_t_furma2']), axis=1)
    grouped_t_furma_df_gas['furma_chosen_max'] = grouped_t_furma_df_gas.apply(
        lambda x: max(x['max_t_furma1'], x['max_t_furma2']), axis=1)

    grouped_t_furma_df_gas = grouped_t_furma_df_gas[['furma_chosen_mean', 'furma_chosen_max']]
    return grouped_t_furma_df_gas


def get_filtered_gas_for_active_furm(df_gas, o2_final_pivot):
    df_gas_filtered = pd.DataFrame()
    for nplv_now in df_gas['NPLV'].unique():
        VR_NACH_now = o2_final_pivot.loc[nplv_now, 'VR_NACH']
        VR_KON_now = o2_final_pivot.loc[nplv_now, 'VR_KON']
        df_gas_now = df_gas[
            (df_gas['NPLV'] == nplv_now) &
            (df_gas['Time'] >= VR_NACH_now) &
            (df_gas['Time'] <= VR_KON_now)]
        df_gas_filtered = df_gas_filtered.append(df_gas_now)
    return df_gas_filtered


def return_gas_volumes(df_gas: pd.DataFrame) -> pd.DataFrame:
    """
    Функция расчета суммарного объема отходящих газов по типам
    и массы вышедшего с газами углерода (Артем)
    """
    gas_list = ['O2', 'N2', 'H2', 'CO2', 'CO', 'AR']
    agg_funcs = {key: 'mean' for key in gas_list}
    agg_funcs['V'] = 'sum'
    result = df_gas.groupby('NPLV').agg(agg_funcs)
    for gas in gas_list:
        result[f'{gas}_volume'] = result[gas] * result['V'] / 100
    result.drop(columns=['V'] + gas_list, inplace=True)
    result.reset_index(inplace=True)
    result['out_C'] = 0.468 * (1.25 * 0.43 * result['CO_volume'] + 1.98 * 0.27 * result['CO2_volume'])
    return result


def time_between_plavki(df_chronom: pd.DataFrame) -> pd.DataFrame:
    """
    Функция расчета времени между плавками (Артем)
    """
    df_chronom.VR_NACH = pd.to_datetime(df_chronom.VR_NACH)
    df_chronom.VR_KON = pd.to_datetime(df_chronom.VR_KON)
    df_chronom['time_betw_plavki'] = (df_chronom.VR_KON - df_chronom.VR_NACH).apply(lambda x: x.seconds)
    result = df_chronom[df_chronom['TYPE_OPER'] == 'межпл.прост.'].groupby('NPLV').sum()
    result = result[['time_betw_plavki']].reset_index()
    return result


def return_gas_heat(df_gas: pd.DataFrame) -> pd.DataFrame:
    """
    Функция расчета интегральной оценки отведенного с газами тепла (Артем)
    """
    df_gas['gas_heat'] = df_gas['T'] * df_gas['V']
    result = df_gas.groupby('NPLV').sum()
    result.reset_index(inplace=True)
    return result[['NPLV', 'gas_heat']]


def time_fault(df_chronom: pd.DataFrame) -> pd.DataFrame:
    """
    Функция расчета аварийных метрик плавки (Артем)
    """
    df_chronom.VR_NACH = pd.to_datetime(df_chronom.VR_NACH)
    df_chronom.VR_KON = pd.to_datetime(df_chronom.VR_KON)
    df_chronom['time_fault'] = (df_chronom.VR_KON - df_chronom.VR_NACH).apply(lambda x: x.seconds)
    time_fault = df_chronom[df_chronom['TYPE_OPER'] == 'вн.пл.прост.'].groupby('NPLV').sum()[['time_fault']]
    time_fault.reset_index(inplace=True)
    result = pd.DataFrame({'NPLV': df_chronom['NPLV'].unique()})
    result = result.merge(time_fault, on='NPLV', how='left').fillna(0.0)

    def fault_type(time_fault: float) -> str:
        if time_fault > 1500:
            return 'long_fault'
        elif time_fault > 0:
            return 'fault'
        else:
            return 'normal'

    result['fault_type'] = result['time_fault'].apply(fault_type)
    return result


def time_between_plavki(df_chronom: pd.DataFrame) -> pd.DataFrame:
    """
    Функция расчета времени между плавками (Артем)
    """
    df_chronom.VR_NACH = pd.to_datetime(df_chronom.VR_NACH)
    df_chronom.VR_KON = pd.to_datetime(df_chronom.VR_KON)
    df_chronom['time_betw_plavki'] = (df_chronom.VR_KON - df_chronom.VR_NACH).apply(lambda x: x.seconds)
    result = df_chronom[df_chronom['TYPE_OPER'] == 'межпл.прост.'].groupby('NPLV').sum()
    result = result[['time_betw_plavki']].reset_index()
    return result
