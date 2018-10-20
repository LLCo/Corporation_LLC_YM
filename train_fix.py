# -*- coding:utf-8 -*-
'''
@Author:

@Date: 

@Description:
    
'''

# 1815101
import pandas as pd
import numpy as np


# train_data = pd.read_table('../data/oppo_round1_train_20180929.txt',
#                            names=['prefix', 'query_prediction', 'title', 'tag', 'label'],
#                            header=None, encoding='utf-8').astype(str)
# error_loc = np.where(train_data['label'].values == '音乐')[0][0]
# print(error_loc)
#
# train_data_dict_df = pd.read_csv('../data/train_vec_4.csv')
# print(train_data_dict_df.iloc[error_loc, :])

# a = pd.DataFrame(np.array([[1,2,3],[1,2,3]]))

data = np.load('../data/dict_pca.npy')
data = np.delete(data, 1815101, axis=0)
print(data.shape)
np.save('train_pca', data[0:1999998])
np.save('val_pca', data[1999998:1999998+50000])
np.save('test_pca', data[1999998+50000:])
