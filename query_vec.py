import gensim as gs
import jieba
import numpy as np
import re
import pandas as pd
import time


r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
path = '../data/word_vec/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
model = gs.models.KeyedVectors.load_word2vec_format(path, binary=True)


def get_vec(word):  # 获得词向量
    if word in model:
        return model.wv[word]
    else:
        return np.zeros(64)
    # try:
    #    return model.wv[word]
    # except KeyError as e:
    #    #print(e)
    #    return np.zeros(64)


def sen_cut(sen):  # jieba分割句子
    jieba.add_word('治好')
    jieba.add_word('五菱宏光')
    jieba.add_word('王境泽')
    se = jieba.lcut(sen, cut_all=False, HMM=True)
    return se


def get_stvec(sen):  # 获得句子向量
    sen = re.sub(r1, '', sen).replace(' ','')
    if len(sen) == 0:
        return np.zeros(64)
    word_l = sen_cut(str(sen))
    vec = []
    for item in word_l:
        vec.append(get_vec(item))
    vec = np.array(vec)
    return vec.sum(axis=0)


def get_dict_vec(dic):  # 获得字典（附加信息）的句向量
    dic = eval(dic)
    sen_l = []
    weight = []
    for item in dic.keys():
        sen_l.append((get_stvec(item)*float(dic[item])))
        weight.append(float(dic[item]))
    length = 10-len(sen_l)
    res = np.zeros((length,64)).tolist()
    weight = np.array(weight)
    #print(res,sen_l)
    res.extend(sen_l)
    sen_l = np.array(res)
    return np.array(sen_l/weight.sum())


def cos(vector1, vector2):
    op7 = np.dot(vector1, vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return op7


def get_similarity(dic0, title):  # 获得字典（附加信息）与title的相似度
    dic = eval(dic0)
    dic_sorted = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    num = np.zeros(10)
    simi = np.zeros(10)
    for (i, item) in enumerate(dic_sorted):
        simi[i] = cos(get_stvec(item[0]), get_stvec(title))
        num[i] = item[1]
    simi = np.nan_to_num(simi)
    num = np.nan_to_num(num)
    mean = simi.mean()
    num_toone = num/num.sum()
    return np.hstack([simi, num, num_toone, mean])


def get_semantic_attr(dataframe):
    attr = []
    s = time.time()
    for index, row in dataframe.iterrows():
        attr.append(get_similarity(row['query_prediction'], row['title_tag']))
        if index % 10000 == 0:
           s_s = time.time()-s
           print('###-itre', index, '-###')
           print('time:', s_s)
           s = time.time()

    length = len(attr[0])
    attr_name = ['new_tag'+str(x) for x in range(length)]
    new_df = pd.DataFrame(attr, columns=attr_name)
    return new_df


