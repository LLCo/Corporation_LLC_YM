# -*- coding:utf-8 -*-
'''
@Author:

@Date: 

@Description:
    
'''

import numpy as np
import pandas as pd


def pre_process(df):
    df.dropna(subset=['title'], inplace=True)


def static_click_count(df):
    '''
    :param df: dataframe use to statistic
    :return:
    '''
    df['assemble'] = df['prefix'] + ' ' + df['title'] + ' ' + df['type']
    df_click = df.loc[df['click'] == 1, :].groupby('assemble', as_index=False)
    df_noclick = df.loc[df['click'] == 0, :].groupby('assemble', as_index=False)
    click = df_click.count().loc[:, ['assemble', 'click']]
    unclick = df_noclick.count().loc[:, ['assemble', 'click']]
    result = pd.merge(click, unclick, how='outer', on=['assemble', 'assemble'])
    result.rename(columns={'click_x': 'enclick', 'click_y': 'unclick'}, inplace=True)
    result.fillna(0, inplace=True)
    result['rate'] = result['enclick'] / (result['enclick'] + result['unclick'])
    result['total'] = result['enclick'] + result['unclick']
    return result


def click_count_assemble(test_df, statistic):
    test_df['assemble'] = test_df['prefix'] + ' ' + test_df['title'] + ' ' + test_df['type']
    test_result = pd.merge(test_df, statistic, left_on='assemble', right_on='assemble', how='left')
    test_result = test_result.loc[:, ['assemble', 'enclick', 'unclick', 'total', 'rate']]
    return test_result


def static_predict(train, test):
    pre_process(train)
    statistic_result = static_click_count(train)
    test_result = click_count_assemble(test, statistic_result)
    test_result.fillna(0, inplace=True)
    return test_result


if __name__ == "__main__":
    test_df = pd.read_csv('..\data\oppo_train_valt_est_csv\oppo_round1_test_A_20180929.csv', encoding='UTF-8')
    train_df = pd.read_csv('..\data\oppo_train_valt_est_csv\oppo_round1_train_20180929.csv', encoding='UTF-8')
    vali_df = pd.read_csv('..\data\oppo_train_valt_est_csv\oppo_round1_vali_20180929.csv', encoding='UTF-8')

    true = vali_df.drop(['click'], axis=1)
    test_result = static_predict(train_df, vali_df)
    rate = test_result.drop(['rate'], axis=1)


    print(test_result.head(5))
    print()
    print(test_result.describe(include='all'))

    pass

