# -*- coding:utf-8 -*-
'''
@Author:

@Date: 

@Description:
    
'''

import gensim as gs
import pandas as pd
import numpy as np
import os
import jieba
import re
import time


'''
Data reading.
'''

# val_data = pd.read_table('../data/oppo_round1_vali_20180929.txt',
#                          names=['prefix', 'query_prediction', 'title', 'tag', 'label'],
#                          header=None, encoding='utf-8').astype(str)
# test_data = pd.read_table('../data/oppo_round1_test_A_20180929.txt',
#                           names=['prefix', 'query_prediction', 'title', 'tag'],
#                           header=None, encoding='utf-8').astype(str)
r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
path = '../data/word_vec/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
word_vec_csv_path = '../data/word_vec_csv.csv'


def get_vec(word, model):  # 获得词向量

    if model is None:
        return np.zeros(64)

    if word in model:
        return model.wv[word]
    else:
        return np.zeros(64)


def sen_cut(sen):  # jieba 分割句子
    sen = re.sub(r1, '', sen.replace(' ', ''))
    se = jieba.lcut(sen, cut_all=False, HMM=True)
    return se


def vector_csv_update(df):
    s = time.time()
    '''
    提取所有sample使用到的词向量特征，保存在csv中。
    :return:
    '''

    column_names = ['word'] + [str(i) for i in range(64)]
    single_word_items = ['prefix', 'title']
    dict_word_item = 'query_prediction'
    jieba.add_word('治好')
    jieba.add_word('五菱宏光')
    jieba.add_word('王境泽')
    jieba.add_word('宋小宝')

    if ~os.path.exists(word_vec_csv_path):
        pd.DataFrame(columns=column_names).to_csv(word_vec_csv_path, index=False)

    word_vec_df = pd.read_csv(word_vec_csv_path)
    row_length = word_vec_df.shape[0]
    w2v_model = gs.models.KeyedVectors.load_word2vec_format(path, binary=True)
    # w2v_model = None
    print('Load Word2Vec Model Successful')

    # temp_datefram = pd.DataFrame(['word'] + list(range(64)), columns=column_names)
    print('Single word start')
    word_and_vec = [0] * 65
    for single_word_item in single_word_items:
        for index, row in df.iterrows():
            '''
            拆出每个单词；
            '''
            sen = sen_cut(row[single_word_item])
            if len(sen_cut(row[single_word_item])) == 0:
                continue
            word_l = sen_cut(str(sen))

            '''
            对每个单词进行W2V操作
            '''
            for word in word_l:
                # 如果该单词已经在word_vec_df中被处理，则不处理它，只处理那些还未处理过的单词。
                if len(pd.Series([])) or ~pd.Series((word_vec_df['word'] == word)).any():
                    word_and_vec[0] = word
                    word_and_vec[1:] = get_vec(word, w2v_model)
                    word_vec_df.loc[row_length] = word_and_vec
                    row_length += 1
            if index % 1000 == 0:
                print('###-itre', index, '-###')

    print('Multiple words start')
    for index, row in df.iterrows():
        for sentence in eval(row[dict_word_item]).keys():
            '''
            拆出每个单词；
            '''
            sen = sen_cut(sentence)
            if len(sen_cut(sentence)) == 0:
                continue
            word_l = sen_cut(str(sen))

            '''
            对每个单词进行W2V操作
            '''

            for word in word_l:
                # 如果该单词已经在word_vec_df中被处理，则不处理它，只处理那些还未处理过的单词。
                a = word_vec_df['word'] == word
                a = a.any()
                if len(pd.Series([])) or ~(word_vec_df['word'] == word).any():
                    word_and_vec[0] = word
                    word_and_vec[1:] = get_vec(word, w2v_model)
                    word_vec_df.loc[row_length] = word_and_vec
                    row_length += 1
            if index % 1000 == 0:
                print('###-itre', index, '-###')

    word_vec_df.to_csv(word_vec_csv_path, index=False)
    print('Update time:', time.time() - s)


def get_feature_dataframe():


if __name__ == "__main__":
    train_data = pd.read_table('../data/oppo_round1_train_20180929.txt',
                               names=['prefix', 'query_prediction', 'title', 'tag', 'label'],
                               header=None, encoding='utf-8').astype(str)
    vector_csv_update(train_data)
