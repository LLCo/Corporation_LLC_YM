# -*- coding:utf-8 -*-
'''
@Author: LLC

@Date: 2018/10/11

@Description: Transform the txt document to csv document.
txt input path: root/data/data_txt
csv output path: root/data/data_csv

The feature name is following:
Prefix, Sup_Dict, Title, Type, Click

Example:
    Fot this kind of item:
    吃鸡	{"吃鸡名字": "0.050", "吃鸡名字大全": "0.010", "吃鸡神器": "0.012", "吃鸡怎么改名字": "0.008", "吃鸡搞笑名字": "0.007",
         "吃鸡网名": "0.011", "吃鸡视频": "0.007", "吃鸡手游": "0.011", "吃鸡游戏": "0.069"}	绝地求生官网	网站

    Prefix: 吃鸡
    Sup_Dict: {"吃鸡名字": "0.050", "吃鸡名字大全": "0.010", "吃鸡神器": "0.012", "吃鸡怎么改名字": "0.008", "吃鸡搞笑名字": "0.007",
               "吃鸡网名": "0.011", "吃鸡视频": "0.007", "吃鸡手游": "0.011", "吃鸡游戏": "0.069"}
    Title: 绝地求生官网
    Type: 网站
    Click: Uncertain (-1)
'''

import os
import pandas as pd
from csv import DictWriter
from numpy import unicode


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


def is_punctuation(uchar):
    """判断一个unicode是否是标点"""
    if uchar >= u'\u0000' and uchar <= u'\u007F':
        return True
    else:
        return False


def format_str(content):
    # content = content.encode()
    content_str = ''
    for i in content:
        if is_chinese(i) or is_number(i) or is_alphabet(i) or is_punctuation(i):
            content_str = content_str + i
    return content_str


def transform(txt_path, csv_path):

    def line_split(line):
        '''
        :param line: sub line
        :return: Prefix, Sup_dict, Title, Type, Click
        '''
        line = format_str(line)
        loc1, loc2 = line.find('{'), line.find('}')
        sup_dict = line[loc1: loc2 + 1]
        prefix = line[:loc1]

        temp_split = line[loc2 + 1:].split()
        if temp_split[-1].isdigit():
            click = int(temp_split[-1])
            type = temp_split[-2]
            title = '. '.join(temp_split[:-2])
        else:
            click = -1
            type = temp_split[-1]
            title = '. '.join(temp_split[:-1])

        return prefix, sup_dict, title, type, click

    features = ['prefix', 'sup_dict', 'title', 'type', 'click']
    with open(txt_path, mode='r', encoding='UTF-8') as f:
        out_f = open(csv_path, 'w', encoding='UTF-8')
        headers = features
        writer = DictWriter(out_f, fieldnames=headers, lineterminator='\n')
        writer.writeheader()
        line = f.readline()
        userFeature_dict = {}
        for feature in features:
            userFeature_dict[feature] = None
        while line:
            arr = line_split(line)
            for i_feature, feature in enumerate(features):
                if i_feature <= 3:
                    userFeature_dict[feature] = arr[i_feature].encode('UTF-8')
                userFeature_dict[feature] = arr[i_feature]
            writer.writerow(userFeature_dict)
            line = f.readline()
        out_f.close()


if __name__ == "__main__":
    input_path1 = '../data/oppo_train_valt_est/oppo_round1_test_A_20180929/oppo_round1_test_A_20180929.txt'
    input_path2 = '../data/oppo_train_valt_est/oppo_round1_train_20180929/oppo_round1_train_20180929.txt'
    input_path3 = '../data/oppo_train_valt_est/oppo_round1_vali_20180929/oppo_round1_vali_20180929.txt'
    output_path1 = '../data/oppo_train_valt_est_csv/oppo_round1_test_A_20180929.csv'
    output_path2 = '../data/oppo_train_valt_est_csv/oppo_round1_train_20180929.csv'
    output_path3 = '../data/oppo_train_valt_est_csv/oppo_round1_vali_20180929.csv'
    transform(input_path1, output_path1)
    transform(input_path2, output_path2)
    transform(input_path3, output_path3)
