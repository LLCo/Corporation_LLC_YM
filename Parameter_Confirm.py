# -*- coding:utf-8 -*-
'''
@Author:

@Date: 10/17

@Description:

线下：
- cross validation score (f1) -: [0.7368719037508845, 0.7356878382227287, 0.7379221889696451, 0.7412807318467696, 0.7243729157604756] . Mean:  0.7352271157101007
- whole validation score (f1) -: [0.7765164351067434, 0.7603833865814696, 0.7713093582356447, 0.7618394352510788, 0.7665115216584335] . Mean:  0.767312027366674

BASE : 0.735075057629144

[{'Max_Depths': 3, 'Num_Leaves': 50, 'Mean_F1': 0.7316891563132697},
 {'Max_Depths': 3, 'Num_Leaves': 80, 'Mean_F1': 0.7316891563132697},
 {'Max_Depths': 3, 'Num_Leaves': 110, 'Mean_F1': 0.7316891563132697},
 {'Max_Depths': 3, 'Num_Leaves': 140, 'Mean_F1': 0.7316891563132697},
 {'Max_Depths': 5, 'Num_Leaves': 50, 'Mean_F1': 0.7326274067886125},
 {'Max_Depths': 5, 'Num_Leaves': 80, 'Mean_F1': 0.7326274067886125},
 {'Max_Depths': 5, 'Num_Leaves': 110, 'Mean_F1': 0.7326274067886125},
 {'Max_Depths': 5, 'Num_Leaves': 140, 'Mean_F1': 0.7326274067886125},
 {'Max_Depths': 7, 'Num_Leaves': 50, 'Mean_F1': 0.7352766260010823},
 {'Max_Depths': 7, 'Num_Leaves': 80, 'Mean_F1': 0.7343603782733515},
 {'Max_Depths': 7, 'Num_Leaves': 110, 'Mean_F1': 0.734183334496089},
 {'Max_Depths': 7, 'Num_Leaves': 140, 'Mean_F1': 0.7340110430495196},
 {'Max_Depths': 9, 'Num_Leaves': 50, 'Mean_F1': 0.7338566871313519},
 {'Max_Depths': 9, 'Num_Leaves': 80, 'Mean_F1': 0.7348625437727156},
 {'Max_Depths': 9, 'Num_Leaves': 110, 'Mean_F1': 0.7319012897056115},
 {'Max_Depths': 9, 'Num_Leaves': 140, 'Mean_F1': 0.733590702315155}]

[{'Max_Depths': 6, 'Num_Leaves': 10, 'Mean_F1': 0.730810569413624},
{'Max_Depths': 6, 'Num_Leaves': 15, 'Mean_F1': 0.7317106990408343},
{'Max_Depths': 6, 'Num_Leaves': 20, 'Mean_F1': 0.7339704115798256},
{'Max_Depths': 6, 'Num_Leaves': 25, 'Mean_F1': 0.7353161867988416},
{'Max_Depths': 6, 'Num_Leaves': 30, 'Mean_F1': 0.7347445447435779},
{'Max_Depths': 6, 'Num_Leaves': 35, 'Mean_F1': 0.7344349932502778},
{'Max_Depths': 6, 'Num_Leaves': 40, 'Mean_F1': 0.7344128489246601},
{'Max_Depths': 6, 'Num_Leaves': 45, 'Mean_F1': 0.7331130280455358},
{'Max_Depths': 6, 'Num_Leaves': 50, 'Mean_F1': 0.734879315424479},
{'Max_Depths': 6, 'Num_Leaves': 55, 'Mean_F1': 0.7338479791124168},
{'Max_Depths': 6, 'Num_Leaves': 60, 'Mean_F1': 0.7346828529137028},
{'Max_Depths': 6, 'Num_Leaves': 65, 'Mean_F1': 0.7346385284250794},
{'Max_Depths': 7, 'Num_Leaves': 10, 'Mean_F1': 0.731883966112812},
{'Max_Depths': 7, 'Num_Leaves': 15, 'Mean_F1': 0.7315034804534699},
{'Max_Depths': 7, 'Num_Leaves': 20, 'Mean_F1': 0.7329085291510112},
{'Max_Depths': 7, 'Num_Leaves': 25, 'Mean_F1': 0.7338504753984748},
{'Max_Depths': 7, 'Num_Leaves': 30, 'Mean_F1': 0.7355227745506892},
{'Max_Depths': 7, 'Num_Leaves': 35, 'Mean_F1': 0.7347819731277754},
{'Max_Depths': 7, 'Num_Leaves': 40, 'Mean_F1': 0.7338613280162013},
{'Max_Depths': 7, 'Num_Leaves': 45, 'Mean_F1': 0.7324465341463433},
{'Max_Depths': 7, 'Num_Leaves': 50, 'Mean_F1': 0.7352766260010823},
{'Max_Depths': 7, 'Num_Leaves': 55, 'Mean_F1': 0.7353947056190444},
{'Max_Depths': 7, 'Num_Leaves': 60, 'Mean_F1': 0.7344384217981996},
{'Max_Depths': 7, 'Num_Leaves': 65, 'Mean_F1': 0.7342523651204736},
{'Max_Depths': 8, 'Num_Leaves': 10, 'Mean_F1': 0.731788229822041},
{'Max_Depths': 8, 'Num_Leaves': 15, 'Mean_F1': 0.7325933861047507},
{'Max_Depths': 8, 'Num_Leaves': 20, 'Mean_F1': 0.7326075499743351},
{'Max_Depths': 8, 'Num_Leaves': 25, 'Mean_F1': 0.7341258854583224},
{'Max_Depths': 8, 'Num_Leaves': 30, 'Mean_F1': 0.733868721571944},
{'Max_Depths': 8, 'Num_Leaves': 35, 'Mean_F1': 0.7339733206941406},
{'Max_Depths': 8, 'Num_Leaves': 40, 'Mean_F1': 0.7353226318798922},
{'Max_Depths': 8, 'Num_Leaves': 45, 'Mean_F1': 0.7344666547720125},
{'Max_Depths': 8, 'Num_Leaves': 50, 'Mean_F1': 0.7348800977795324},
{'Max_Depths': 8, 'Num_Leaves': 55, 'Mean_F1': 0.7349206600524768},
{'Max_Depths': 8, 'Num_Leaves': 60, 'Mean_F1': 0.7348416130247999},
{'Max_Depths': 8, 'Num_Leaves': 65, 'Mean_F1': 0.7345746248312139}]


[{'Max_Depths': 7, 'Num_Leaves': 30, 'Mean_F1': 0.7348542440245853},
{'Max_Depths': 7, 'Num_Leaves': 33, 'Mean_F1': 0.7343976171772876},
{'Max_Depths': 7, 'Num_Leaves': 36, 'Mean_F1': 0.7347729022080465},
{'Max_Depths': 7, 'Num_Leaves': 39, 'Mean_F1': 0.7337995993212668},
{'Max_Depths': 7, 'Num_Leaves': 42, 'Mean_F1': 0.7360028091940335},
{'Max_Depths': 7, 'Num_Leaves': 45, 'Mean_F1': 0.7344552141393473},
{'Max_Depths': 7, 'Num_Leaves': 48, 'Mean_F1': 0.7354584694394919},
{'Max_Depths': 7, 'Num_Leaves': 51, 'Mean_F1': 0.7337081206063925},
{'Max_Depths': 7, 'Num_Leaves': 54, 'Mean_F1': 0.7346611763778446},
{'Max_Depths': 7, 'Num_Leaves': 57, 'Mean_F1': 0.7351101969024281},
{'Max_Depths': 7, 'Num_Leaves': 60, 'Mean_F1': 0.7334746152021143},
{'Max_Depths': 7, 'Num_Leaves': 63, 'Mean_F1': 0.7345330079477936},
{'Max_Depths': 7, 'Num_Leaves': 66, 'Mean_F1': 0.7352694218855781},
{'Max_Depths': 7, 'Num_Leaves': 69, 'Mean_F1': 0.734799929539864},
{'Max_Depths': 7, 'Num_Leaves': 72, 'Mean_F1': 0.7348402135410655},
{'Max_Depths': 7, 'Num_Leaves': 75, 'Mean_F1': 0.7340515676912834},
{'Max_Depths': 7, 'Num_Leaves': 78, 'Mean_F1': 0.7341561008321588},
{'Max_Depths': 7, 'Num_Leaves': 81, 'Mean_F1': 0.7341117405115549},
{'Max_Depths': 7, 'Num_Leaves': 84, 'Mean_F1': 0.7335144094092318},
{'Max_Depths': 7, 'Num_Leaves': 87, 'Mean_F1': 0.73297839417144},
{'Max_Depths': 7, 'Num_Leaves': 90, 'Mean_F1': 0.7345015712588865},
{'Max_Depths': 7, 'Num_Leaves': 93, 'Mean_F1': 0.7352015569980093},
{'Max_Depths': 7, 'Num_Leaves': 96, 'Mean_F1': 0.7342819754650672},
{'Max_Depths': 7, 'Num_Leaves': 99, 'Mean_F1': 0.7336886290111709},
{'Max_Depths': 7, 'Num_Leaves': 102, 'Mean_F1': 0.734056909519752},
{'Max_Depths': 7, 'Num_Leaves': 105, 'Mean_F1': 0.7342451033123008},
{'Max_Depths': 7, 'Num_Leaves': 108, 'Mean_F1': 0.7346916982084675},
{'Max_Depths': 7, 'Num_Leaves': 111, 'Mean_F1': 0.73473116715507},
{'Max_Depths': 7, 'Num_Leaves': 114, 'Mean_F1': 0.735130751831439},
{'Max_Depths': 7, 'Num_Leaves': 117, 'Mean_F1': 0.7346790083686654},
{'Max_Depths': 7, 'Num_Leaves': 120, 'Mean_F1': 0.7338931738445347},
{'Max_Depths': 7, 'Num_Leaves': 123, 'Mean_F1': 0.7345864231560648},
{'Max_Depths': 7, 'Num_Leaves': 126, 'Mean_F1': 0.7344530128413678},
{'Max_Depths': 7, 'Num_Leaves': 129, 'Mean_F1': 0.7351234660620755},
{'Max_Depths': 7, 'Num_Leaves': 132, 'Mean_F1': 0.7351234660620755},
{'Max_Depths': 7, 'Num_Leaves': 135, 'Mean_F1': 0.7351234660620755},
{'Max_Depths': 7, 'Num_Leaves': 138, 'Mean_F1': 0.7351234660620755},
{'Max_Depths': 7, 'Num_Leaves': 141, 'Mean_F1': 0.7351234660620755},
{'Max_Depths': 7, 'Num_Leaves': 144, 'Mean_F1': 0.7351234660620755},
{'Max_Depths': 7, 'Num_Leaves': 147, 'Mean_F1': 0.7351234660620755}]

[{'Max_Depths': 7, 'Num_Leaves': 30, 'Mean_F1': 0.7351772845194388}, {'Max_Depths': 7, 'Num_Leaves': 35, 'Mean_F1': 0.7359350435078639}, {'Max_Depths': 7, 'Num_Leaves': 40, 'Mean_F1': 0.7359923805275593}, {'Max_Depths': 7, 'Num_Leaves': 45, 'Mean_F1': 0.7367862396953377}, {'Max_Depths': 7, 'Num_Leaves': 50, 'Mean_F1': 0.7363514637580938}, {'Max_Depths': 7, 'Num_Leaves': 55, 'Mean_F1': 0.7364623731126875}, {'Max_Depths': 7, 'Num_Leaves': 60, 'Mean_F1': 0.7359169662574153}, {'Max_Depths': 7, 'Num_Leaves': 65, 'Mean_F1': 0.7371471741935071}, {'Max_Depths': 7, 'Num_Leaves': 70, 'Mean_F1': 0.7360352934732435}, {'Max_Depths': 7, 'Num_Leaves': 75, 'Mean_F1': 0.7365993937175279}, {'Max_Depths': 7, 'Num_Leaves': 80, 'Mean_F1': 0.7364845250348659}, {'Max_Depths': 7, 'Num_Leaves': 85, 'Mean_F1': 0.7364556844697588}, {'Max_Depths': 7, 'Num_Leaves': 90, 'Mean_F1': 0.7369094348392805}, {'Max_Depths': 7, 'Num_Leaves': 95, 'Mean_F1': 0.7367760386365239}, {'Max_Depths': 7, 'Num_Leaves': 100, 'Mean_F1': 0.7361546304052251}, {'Max_Depths': 7, 'Num_Leaves': 105, 'Mean_F1': 0.7364417969557636}, {'Max_Depths': 7, 'Num_Leaves': 110, 'Mean_F1': 0.7377116988908029}, {'Max_Depths': 7, 'Num_Leaves': 115, 'Mean_F1': 0.7370063105423759}, {'Max_Depths': 7, 'Num_Leaves': 120, 'Mean_F1': 0.7360298918065817}, {'Max_Depths': 7, 'Num_Leaves': 125, 'Mean_F1': 0.7362777340089784}, {'Max_Depths': 7, 'Num_Leaves': 130, 'Mean_F1': 0.7356851393211343}, {'Max_Depths': 7, 'Num_Leaves': 135, 'Mean_F1': 0.7356851393211343}, {'Max_Depths': 7, 'Num_Leaves': 140, 'Mean_F1': 0.7356851393211343}, {'Max_Depths': 7, 'Num_Leaves': 145, 'Mean_F1': 0.7356851393211343}]



'''

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def lgb_f1_score_sk(y_hat, y_true):
    y_true = np.round(y_true)
    y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True


'''
Initial Configuration
'''
pd.set_option('display.expand_frame_repr', False)

'''
Data reading.
'''
train_data = pd.read_table('../data/oppo_round1_train_20180929.txt',
                           names=['prefix', 'query_prediction', 'title', 'tag', 'label'],
                           header=None, encoding='utf-8').astype(str)
val_data = pd.read_table('../data/oppo_round1_vali_20180929.txt',
                         names=['prefix', 'query_prediction', 'title', 'tag', 'label'],
                         header=None, encoding='utf-8').astype(str)
test_data = pd.read_table('../data/oppo_round1_test_A_20180929.txt',
                          names=['prefix', 'query_prediction', 'title', 'tag'],
                          header=None, encoding='utf-8').astype(str)

'''
Data preprocessing (clearing)
'''

train_data['label'] = train_data['label'].apply(lambda x: int(x))
val_data['label'] = val_data['label'].apply(lambda x: int(x))

'''
Feature Enginnering
'''

items = ['prefix', 'title', 'tag']
temp = train_data.groupby(items, as_index=False)['label'].agg(
    {'_'.join(items) + '_click': 'sum', '_'.join(items) + '_count': 'count'})
temp['_'.join(items) + '_ctr'] = temp['_'.join(items) + '_click'] / (temp['_'.join(items) + '_count'])
train_data = pd.merge(train_data, temp, on=items, how='left')
val_data = pd.merge(val_data, temp, on=items, how='left')
test_data = pd.merge(test_data, temp, on=items, how='left')

for item in items:
    temp = train_data.groupby(item, as_index=False)['label'].agg({item + '_click': 'sum', item + '_count': 'count'})
    temp[item + '_ctr'] = temp[item + '_click'] / (temp[item + '_count'])
    train_data = pd.merge(train_data, temp, on=item, how='left')
    val_data = pd.merge(val_data, temp, on=item, how='left')
    test_data = pd.merge(test_data, temp, on=item, how='left')

for i in range(len(items)):
    for j in range(i + 1, len(items)):
        item_g = [items[i], items[j]]
        temp = train_data.groupby(item_g, as_index=False)['label'].agg(
            {'_'.join(item_g) + '_click': 'sum', '_'.join(item_g) + '_count': 'count'})
        temp['_'.join(item_g) + '_ctr'] = temp['_'.join(item_g) + '_click'] / (temp['_'.join(item_g) + '_count'])
        train_data = pd.merge(train_data, temp, on=item_g, how='left')
        val_data = pd.merge(val_data, temp, on=item_g, how='left')
        test_data = pd.merge(test_data, temp, on=item_g, how='left')

# semantic feature - 31_3

train_data = pd.concat([train_data, pd.read_csv('../data/train_31_3.csv')], axis=1)
val_data = pd.concat([val_data, pd.read_csv('../data/val_31_3.csv')], axis=1)
test_data = pd.concat([test_data, pd.read_csv('../data/test_31_3.csv')], axis=1)

# semantic feature - 4

train_data_dict_df = pd.read_csv('../data/train_vec_4.csv')
val_data_dict_df = pd.read_csv('../data/val_vec_4.csv')
test_data_dict_df = pd.read_csv('../data/test_vec_4.csv')
train_data = pd.concat([train_data, train_data_dict_df], axis=1)
val_data = pd.concat([val_data, val_data_dict_df], axis=1)
test_data = pd.concat([test_data, test_data_dict_df], axis=1)

# one-hot of 'tag'

train_data = pd.get_dummies(train_data, columns=['tag']).drop(['tag_推广'], axis=1)
val_data = pd.get_dummies(val_data, columns=['tag']).drop(['tag_推广'], axis=1)
test_data = pd.get_dummies(test_data, columns=['tag'])

# drop useless feature

train_data_ = train_data.drop(['prefix', 'query_prediction', 'title'], axis=1)
val_data_ = val_data.drop(['prefix', 'query_prediction', 'title'], axis=1)
test_data_ = test_data.drop(['prefix', 'query_prediction', 'title'], axis=1)

# similarity of prefix and  title

train_data_ = pd.concat([train_data_, pd.read_csv('../data/train_prefix_1.csv')], axis=1)
val_data_ = pd.concat([val_data_, pd.read_csv('../data/val_prefix_1.csv')], axis=1)
test_data_ = pd.concat([test_data_, pd.read_csv('../data/test_prefix_1.csv')], axis=1)


'''
Training
'''
print('Feature: ', train_data_.columns.values)
print('- Nan Check! -')
print('train_data_:\n', train_data_.isna().sum(axis=0))
print('val_data_:\n', val_data_.isna().sum(axis=0))
print('test_data_:\n', test_data_.isna().sum(axis=0))

# Label Split
X_train_data_ = np.array(train_data_.drop(['label'], axis=1))
y_train_data_ = np.array(train_data_['label'])
X_val_data_ = np.array(val_data_.drop(['label'], axis=1))
y_val_data_ = np.array(val_data_['label'])
X_test_data = test_data_

# Data inspecting
print('train beginning')
print('================================')

print('-Training- : ')
print(X_train_data_.shape)
print(y_train_data_.shape)

print('-Training- : ')
print(X_val_data_.shape)
print(y_val_data_.shape)

print('-Testing- : ')
print(X_test_data.shape)
print('================================')

# Algorithm: LightGBM
N = 5
skf = StratifiedKFold(n_splits=N, random_state=86, shuffle=True)
xx_f1 = []
xx_submit = []
valid_f1 = []

# 先调整Max_Depth and Num_Leaves

Max_Depths = [7]
Num_Leaves = range(30, 150, 5)
result_template = {'Max_Depths': 0, 'Num_Leaves': 0, 'Mean_F1': 0}
Result = []

for max_depth in Max_Depths:
    for num_leave in Num_Leaves:
        result_template['Max_Depths'] = max_depth
        result_template['Num_Leaves'] = num_leave
        xx_f1 = []
        LGBM_classify = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=num_leave, max_depth=max_depth,
                                           learning_rate=0.1, subsample_freq=5, n_estimators=1000, silent=True)
        LGBM_classify.set_params(**{'objective': 'huber', 'metric': 'None'})
        for k, (train_loc, test_loc) in enumerate(skf.split(X_val_data_, y_val_data_)):
            print('train _K_ flod', k)
            X_train_combine = np.vstack([X_train_data_, X_val_data_[train_loc]])
            Y_train_combine = np.hstack([y_train_data_, y_val_data_[train_loc]])

            LGBM_classify.fit(X_train_combine, Y_train_combine, verbose=False,
                              eval_set=(X_val_data_[test_loc], y_val_data_[test_loc]),
                              early_stopping_rounds=150, eval_sample_weight=None, eval_metric=lgb_f1_score_sk)
            xx_f1.append(f1_score(y_val_data_[test_loc], LGBM_classify.predict(X_val_data_[test_loc])))
        result_template['Mean_F1'] = np.mean(xx_f1)
        print(result_template)
        Result.append(result_template.copy())

print('========================')
print('Parameter tune finished!')
print(Result)
print('========================')
