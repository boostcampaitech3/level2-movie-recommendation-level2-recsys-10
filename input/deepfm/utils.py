import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


def neg_sample_popular(popular_item_list, item_set, num_neg = 50,
 neg_sample_by_pos_sample = False, popular_rate = 0.3):
    """
    Args:
        popular_item_list (list): 평가된 횟수 순으로 정렬된 list
        item_set (set): 유저의 item set
        num_neg (int): 뽑을 neg_sample 개수 
        neg_sample_by_pos_sample (boolean): Positive와 동일한 수의 negtive sampling을 할지 여부
        popular_rate (float): 상위 몇 %의 item을 중에 선택할 것인가 결정하는 float

    return neg_items : 선택된 item id들의 list
    """
    popular_rate = int(len(popular_item_list)*popular_rate) 
    popular_item_list = popular_item_list[:popular_rate]
    popular_item_list = list(set(popular_item_list).difference(item_set))
    
    if neg_sample_by_pos_sample:
        num_neg = len(item_set)

    neg_items = np.random.choice(popular_item_list, num_neg, replace=False)

    return neg_items

###### metric ######
def recall(pred, target):
    tp, fn = 0,0
    for p, t in zip(pred,target):
        if p == 1 and t==1 :
            tp += 1
        elif p == 0 and t==1 :
            fn += 1
    return tp / (tp + fn + 1e-10)

def precision(pred, target):
    tp, fp = 0, 0
    for p, t in zip(pred,target):
        if p == 1 and t==1 :
            tp += 1
        elif p == 1 and t==0 :
            fp += 1
    return tp / (tp + fp + 1e-10)

def f1_score(pred, target):
    p_score=precision(pred, target)
    r_score = recall(pred, target)

    return 2*(p_score * r_score) / (p_score + r_score)

def recall(pred, target):
    tp, fn = 0,0
    for i,p in enumerate(pred):
        if p == 1 and target[i] == 1 :
            tp += 1
        elif p == 1 and target[i] == 0:
            fn += 1
    return tp / (tp + fn + 1e-10)

def precision(pred, target):
    tp, fp = 0, 0
    for i,p in enumerate(pred):
        if p == 1 and target[i] == 1 :
            tp += 1
        elif p == 0 and target[i] == 1:
            fp += 1
    return tp / (tp + fp + 1e-10)

def f1_score(pred, target):
    p_score=precision(pred, target)
    r_score = recall(pred, target)
    return 2*(p_score * r_score) / (p_score + r_score +  1e-10), p_score,r_score
