# -*- coding:utf-8 -*-
'''
@Author:

@Date: 10/17

@Description:

线下：
- cross validation score (f1) -: [0.7368719037508845, 0.7356878382227287, 0.7379221889696451, 0.7412807318467696, 0.7243729157604756] . Mean:  0.7352271157101007
- whole validation score (f1) -: [0.7765164351067434, 0.7603833865814696, 0.7713093582356447, 0.7618394352510788, 0.7665115216584335] . Mean:  0.767312027366674

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
X_test_data = np.array(test_data_)

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

# delete parameters

# i = 28
# train_data_ = np.delete(X_train_data_, i, axis=1)
# val_data_ = np.delete(X_val_data_, i, axis=1)
# test_data_ = np.delete(X_test_data, i, axis=1)


# Algorithm: LightGBM
N = 5
skf = StratifiedKFold(n_splits=N, random_state=86, shuffle=True)
xx_f1 = []
xx_submit = []
valid_f1 = []
LGBM_classify = lgb.LGBMClassifier(boosting_type='gbdt', objective='huber', num_leaves=32,
                                   learning_rate=0.05, subsample_freq=5, n_estimators=5000, silent=True)
LGBM_classify.set_params(**{'objective': 'huber', 'metric': 'None'})
LGBM_classify.fit(X_train_data_, y_train_data_,
                  eval_set=(X_val_data_, y_val_data_),
                  early_stopping_rounds=100, eval_sample_weight=None, eval_metric=lgb_f1_score_sk)
best_f1 = LGBM_classify._best_score['valid_0']['f1']

print('ReSet!')
print('\n\n- cross validation score (f1) -:', best_f1)

'''
Save result
'''
# s = 0
# for i in xx_submit:
#     s = s + i
# test_data_['pred_label'] = list(s[:, 1] / N)  # 二元分类中，概率分布对应了- 0，1 -
# test_data_['pred_label'] = test_data_['pred_label'].apply(lambda x: round(x))
# test_data_['pred_label'].to_csv('../data/Result_LLC_1020_pre ' + str(np.mean(xx_f1)) + '.csv', index=False)