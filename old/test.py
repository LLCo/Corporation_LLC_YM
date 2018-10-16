# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:23:31 2018

@author: ym
"""

f=open('..\data\oppo_train_valt_est_csv\oppo_round1_train_20180929.csv').readlines()
s=[]
for line in f:
    s.append(line.split(','))
dic={}
for item in s[1:]:
    if item[-3]+item[-2] not in dic.keys():
        dic[item[-3]+item[-2]]=[0,0]
    else:
        if item[-1]=='0\n':
            dic[item[-3]+item[-2]][0]+=1
        else:
            dic[item[-3]+item[-2]][1]+=1

f = open('..\data\oppo_train_valt_est_csv\oppo_round1_vali_20180929.csv').readlines()
s = []
for line in f:
    s.append(line.split(','))
count = [0, 0]
for item in s[1:]:
    if item[-3]+item[-2] in dic.keys():
        pre = int(dic[item[-3]+item[-2]][1]>dic[item[-3]+item[-2]][0])
    else:
        pre = 0
    if pre == int(item[-1].strip()):
        count[0]+=1
    count[1]+=1
print(count)
