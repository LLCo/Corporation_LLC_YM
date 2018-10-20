# -*- coding:utf-8 -*-
'''
@Author:

@Date: 10/17

@Description:

线下：
- cross validation score (f1) -: [0.7332195676905575, 0.7368121711929474, 0.7399800313792612, 0.7402114889968563, 0.7185088102519296] . Mean:  0.7337464139023104
- whole validation score (f1) -: [0.7663748010006822, 0.7620647439000084, 0.7687327862801329, 0.7631556405790481, 0.7624692206379202] . Mean:  0.7645594384795584

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

train_data_dict_df = pd.read_csv('../data/train_vec_4_no_fix.csv')
val_data_dict_df = pd.read_csv('../data/val_vec_4.csv')
test_data_dict_df = pd.read_csv('../data/test_vec_4.csv')
dict_columns = ['ws_similarity', 'maximum_similarity', 'median_similarity', 'mean_similarity']
for item in dict_columns:
    train_data[item] = train_data_dict_df[item]
    val_data[item] = val_data_dict_df[item]
    test_data[item] = test_data_dict_df[item]

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
skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
xx_f1 = []
xx_submit = []
valid_f1 = []
LGBM_classify = lgb.LGBMClassifier(boosting_type='gbdt', objective='huber', num_leaves=32,
                                   learning_rate=0.05, subsample_freq=5, n_estimators=5000, silent=False)

for k, (train_loc, test_loc) in enumerate(skf.split(X_val_data_, y_val_data_)):
    print('train _K_ flod', k)
    X_train_combine = np.vstack([X_train_data_, X_val_data_[train_loc]])
    Y_train_combine = np.hstack([y_train_data_, y_val_data_[train_loc]])

    LGBM_classify.fit(X_train_combine, Y_train_combine,
                      eval_set=(X_val_data_[test_loc], y_val_data_[test_loc]),
                      early_stopping_rounds=200, eval_sample_weight=None, eval_metric=lgb_f1_score_sk)

    xx_f1.append(LGBM_classify._best_score['valid_0']['f1'])
    xx_submit.append(LGBM_classify.predict_proba(X_test_data, num_iteration=LGBM_classify.best_iteration_))
    valid_f1.append(f1_score(y_val_data_, LGBM_classify.predict(X_val_data_)))

print('ReSet!')
print('\n\n- cross validation score (f1) -:', xx_f1, '. Mean: ', np.mean(xx_f1))
print('- whole validation score (f1) -:', valid_f1, '. Mean: ', np.mean(valid_f1))

'''
Save result
'''
s = 0
for i in xx_submit:
    s = s + i
test_data_['pred_label'] = list(s[:, 1] / N)  # 二元分类中，概率分布对应了- 0，1 -
test_data_['pred_label'] = test_data_['pred_label'].apply(lambda x: round(x))
test_data_['pred_label'].to_csv('../data/Result_LLC_1017_pre ' + str(np.mean(xx_f1)) + '.csv', index=False)
