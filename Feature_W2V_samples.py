# -*- coding:utf-8 -*-
'''
@Author:

@Date: 

@Description:
    
'''

import gensim as gs
import jieba
import numpy as np
import re
import pandas as pd
import time
import query_vec


r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
path = '../data/word_vec/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
model = gs.models.KeyedVectors.load_word2vec_format(path, binary=True)


def get_stvec_llc(sen):  # 获取句向量的语义。
    sen = re.sub(r1, '', sen).replace(' ','')
    if len(sen) == 0:
        return np.zeros(64)
    word_l = query_vec.sen_cut(str(sen))
    word_l_length = len(word_l)

    vec = 0
    for item in word_l:
        vec = query_vec.get_vec(item) / word_l_length
    return vec


def cos(vector1, vector2):  # 获取两个向量的cos夹角
    op7 = np.dot(vector1, vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return op7


def dict_title_semantic(dic0, title):
    '''
    提取dic和title的语义特征，分别为
    1. dic各个句向量相似性的加权和 与 title句向量 的相似性。
    2. dic各个句向量的相似性 与 title句向量 的相似性的最大值。
    3. dic各个句向量的相似性 与 title句向量 的相似性的均值。
    4. dic各个句向量的相似性 与 title句向量 的相似性的中位数。
    # 5. dic各个句向量相似性的加权和。
    :return:
    '''

    sens, weight = list(eval(dic0).keys()), list(eval(dic0).values())
    weight = np.array(weight, dtype=np.float64).reshape(-1, 1)
    if len(sens) == 0:
        return None
    dic_semantic = np.zeros((len(sens), 64))

    for i, sen in enumerate(sens):
        dic_semantic[i, :] = get_stvec_llc(sen)
    title_semantic = get_stvec_llc(title)

    # 计算{}句向量的加权均值，以及其与title向量的相似性
    dic_semantic_weight_sum = np.sum(dic_semantic * (weight / np.sum(weight)), axis=0)
    ws_similarity = cos(title_semantic, dic_semantic_weight_sum)
    ws_similarity = np.nan_to_num(ws_similarity)

    # 计算{}句向量的与title向量的相似性的最大值, 中位数， 均值
    similaritys = np.empty(len(sens))
    for i in range(len(sens)):
        similaritys[i] = cos(title_semantic, dic_semantic[i, :])
    similaritys = np.nan_to_num(similaritys)
    maximum_similarity = np.max(similaritys)
    median_similarity = np.median(similaritys)
    mean_similarity = np.mean(similaritys)

    '''
    4 feature:
    '''
    result = np.empty(4)
    result[:] = ws_similarity, maximum_similarity, median_similarity, mean_similarity
    # result[4:] = dic_semantic_weight_sum
    return result


def get_semantic_df(dataframe):  # 获取到对应额 dataframe 的语义特征 datafram
    s = time.time()
    length = dataframe.shape[0]

    semantic_df = pd.DataFrame(np.empty((length, 4)),
                               columns=['ws_similarity', 'maximum_similarity',
                                        'median_similarity', 'mean_similarity'])
    for index, row in dataframe.iterrows():
        semantic_df.iloc[index, :] = dict_title_semantic(row['query_prediction'], row['title'])
        if index % 10000 == 0:
           s_s = time.time()-s
           print('###-itre', index, '-###')
           print('time:', s_s)
           s = time.time()
    return semantic_df


if __name__ == "__main__":

    test_data = pd.read_table('../data/oppo_round1_test_A_20180929.txt',
                              names=['prefix', 'query_prediction', 'title', 'tag'],
                              header=None, encoding='utf-8').astype(str)
    get_semantic_df(test_data).to_csv('../data/test_vec_4+64.csv', index=False)

    train_data = pd.read_table('../data/oppo_round1_train_20180929.txt',
                               names=['prefix', 'query_prediction', 'title', 'tag', 'label'],
                               header=None, encoding='utf-8').astype(str)
    get_semantic_df(train_data).to_csv('../data/train_vec_4+64.csv', index=False)


    val_data = pd.read_table('../data/oppo_round1_vali_20180929.txt',
                             names=['prefix', 'query_prediction', 'title', 'tag', 'label'],
                             header=None, encoding='utf-8').astype(str)
    get_semantic_df(val_data).to_csv('../data/val_vec_4+64.csv', index=False)
