# -*- coding:utf-8 -*-
'''
@Author:

@Date:

@Description:

'''
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


Result_LLC_1013 = pd.read_csv('../Result_LLC_1013.csv')
result_XXY_ori = pd.read_csv('../result_XXY_ori.csv')

