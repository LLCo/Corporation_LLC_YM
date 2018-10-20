# -*- coding:utf-8 -*-
'''
@Author:

@Date: 10/17

@Description:

线下：
    [[0, 0.7342110171378069],
    [1, 0.735095796758179],
    [2, 0.7343124419724265],
    [3, 0.7345937685954411],
    [4, 0.7347719980842504],
    [5, 0.7341680903301149],
    [6, 0.7351647788937591],
    [7, 0.7356342611204518],
    [8, 0.7339482046619583],
    [9, 0.7348528599701908]
    [10, 0.7349170084286083],
    [11, 0.7335869377211381],
    [12, 0.7352034565897677],
    [13, 0.7344504700926984],
    [14, 0.7347105255844545],
    [15, 0.7345048140481756],
    [16, 0.7349279154563518],
    [17, 0.733877366224241],
    [18, 0.7347214197553587],
    [19, 0.7333751092906651],
    [20, 0.7331635428699361],
    [21, 0.7342652546244786],
    [22, 0.7346621602870758],
    [23, 0.7359203219346563],
    [24, 0.7359659758214004],
    [25, 0.7339109544569269],
    [26, 0.7342344817307639],
    [27, 0.7344534103223003],
    [28, 0.736005707528894],
    [29, 0.7354806141257892],
    [30, 0.7346371737328774],
    [31, 0.7342267153580663],
    [32, 0.7346503113523919],
    [33, 0.7345556676692586],
    [34, 0.7352410437902683],
    [35, 0.7347079364875044],
    [36, 0.7345663204655432],
    [37, 0.7340996273423068],
    [38, 0.7332522244898766],
    [39, 0.7342785027468304],
    [40, 0.7346247252002444],
    [41, 0.735200323953778],
    [42, 0.734337044183019],
    [43, 0.7351520247339205],
    [44, 0.7351355908151976],
    [45, 0.7341802075542874],
    [46, 0.7342207790857758],
    [47, 0.7353048191340179],
    [48, 0.734360909625644],
    [49, 0.7352878415006728],
    [50, 0.734418833639676],
    [51, 0.735540910860702],
    [52, 0.7342928763363468],
    [53, 0.7337331346020206],
    [54, 0.7336562822831266],
    [55, 0.7344096141296822],
    [56, 0.7349319866932646],
    [57, 0.7344338347277078],
    [58, 0.7349975937867342],
    [59, 0.7349927782760057],
    [60, 0.7349674379232018],
    [61, 0.7347423576556322],
    [62, 0.7344081293908794],
    [63, 0.7342433376576972],
    [64, 0.73486384170368],
    [65, 0.7347007239888815],
    [66, 0.7352271157101007],
    [67, 0.7345934933092434],
    [68, 0.7352620621688708],
    [69, 0.7352271157101007],
    [70, 0.7352271157101007],
    [71, 0.7348184656570902],
    [72, 0.7340263958831683],
    [73, 0.7349689783550438],
    [74, 0.7350621374319148],
    [75, 0.7352271157101007],
    [76, 0.734989659168032],
    [77, 0.7352271157101007],
    [78, 0.7341789435010881],
    [79, 0.7333824918908917],
    [80, 0.7306983721685867]]

线上： 0.7294

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
X_test_data_ = np.array(test_data_)

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
print(X_test_data_.shape)
print('================================')

# Algorithm: LightGBM
N = 5
skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
xx_f1 = []
xx_submit = []
valid_f1 = []
LGBM_classify = lgb.LGBMClassifier(boosting_type='gbdt', objective='huber', num_leaves=32,
                                   learning_rate=0.05, subsample_freq=5, n_estimators=5000, silent=True)
result = []
for i in range(X_train_data_.shape[1]):
    if i < 10:
        continue
    X_train_data_temp = np.delete(X_train_data_, i, axis=1)
    X_val_data_temp = np.delete(X_val_data_, i, axis=1)
    X_test_data_temp = np.delete(X_test_data_, i, axis=1)

    for k, (train_loc, test_loc) in enumerate(skf.split(X_val_data_temp, y_val_data_)):
        print('train _K_ flod', k)
        X_train_combine = np.vstack([X_train_data_temp, X_val_data_temp[train_loc]])
        Y_train_combine = np.hstack([y_train_data_, y_val_data_[train_loc]])

        LGBM_classify.fit(X_train_combine, Y_train_combine,
                          eval_set=(X_val_data_temp[test_loc], y_val_data_[test_loc]), verbose=False,
                          early_stopping_rounds=200, eval_sample_weight=None, eval_metric=lgb_f1_score_sk)
        xx_f1.append(LGBM_classify._best_score['valid_0']['f1'])
        xx_submit.append(LGBM_classify.predict_proba(X_test_data_temp, num_iteration=LGBM_classify.best_iteration_))
        # valid_f1.append(f1_score(y_val_data_, LGBM_classify.predict(X_val_data_temp)))
    result.append([i, np.mean(xx_f1)])
    xx_f1 = []
    print(result)

# print('ReSet!')
# print('\n\n- cross validation score (f1) -:', xx_f1, '. Mean: ', np.mean(xx_f1))
# print('- whole validation score (f1) -:', valid_f1, '. Mean: ', np.mean(valid_f1))
# print('!=========================!')
# print(result)
# print('!=========================!')
#
# '''
# Save result
# '''
# s = 0
# for i in xx_submit:
#     s = s + i
# test_data_['pred_label'] = list(s[:, 1] / N)  # 二元分类中，概率分布对应了- 0，1 -
# test_data_['pred_label'] = test_data_['pred_label'].apply(lambda x: round(x))
# test_data_['pred_label'].to_csv('../data/Result_LLC_1017_pre ' + str(np.mean(xx_f1)) + '.csv', index=False)
