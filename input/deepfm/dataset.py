import random
import os
from statistics import mode 

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import neg_sample


class DeepFMDataset(Dataset):
    def __init__(self, rating, year, genre, valid_size=10, mode = 'static'):
        self.mode = mode
        self.valid_size = valid_size

        self.rating_df =rating
        self.year_dict = year
        self.genres_dict = genre
    
        self.n_user = self.rating_df['user'].nunique()
        self.n_item = self.rating_df['item'].nunique()
        
        self.n_year = len(set(self.year_dict.values()))
        self.users = self.rating_df['user'].unique()
        self.rating_df = rating.groupby('user')['item'].agg(lambda x : list(x))
        print("data load done")

    def get_num_context(self):
        return self.n_user, self.n_item, self.n_year
    def pos_or_neg(self):
        return np.random.rand()

    def __len__(self):
        return self.n_user
        
    def __getitem__(self,user_id):
        if self.pos_or_neg() > 0.5: # positive
            if self.mode == 'seq':
                pos_set = self.rating_df[user_id][-self.valid_size:] # 뒤에꺼
            elif self.mode == 'static': # static
                if self.valid_size >= len(self.rating_df[user_id]):
                    pos_set = self.rating_df[user_id][:-10]
                else : pos_set = self.rating_df[user_id][:-self.valid_size]  # 앞에꺼
            self.item_selected = np.random.choice(pos_set)
            self.target = np.array(1)

        else: # negative
            pos_set = self.rating_df[user_id]
            self.item_selected = np.random.randint(1, self.n_item - 1)
            while self.item_selected in pos_set:
                self.item_selected = np.random.randint(1, self.n_item - 1)
            self.target = np.array(0)
            
        # self.item_selected = self.item_selected
        genre_selected = self.genres_dict[self.item_selected]
        year_selected = self.year_dict[self.item_selected]
        self.input = np.array([user_id, self.item_selected, year_selected, *genre_selected])
        return self.input, self.target


class DeepFMDataset_renew(Dataset):
    def __init__(self, rating, year, genre, director, valid_size=10, mode = 'static', train_all = True):
        """
        Args :
            
            - valid_size : 뒤에서 몇개를 잘라 valid에 사용할지 결정
            - mode : seq(시간상 뒤에 valid_size 만큼만 학습) or static(시간상 앞에 valid_size 제외한 만큼 학습)
            - train_all : True => 모델에서 모든 input(user,item,year,genre,director)를 임베딩에 활용
                          False => 모델에서 아이템을 제외한 input(user,year,genre,director)를 임베딩에 활용하여 유저 고유의 성질 파악
            - rating, year, genre, director : 전처리된 데이터프레임 or Dictionary
        """
        self.mode = mode
        self.valid_size = valid_size
        self.train_all = train_all

        # preprocessing 된 데이터를 받아옴 / 아이템과 Join을 위해 year,genre,director는 dict 형태
        self.rating_df =rating
        self.year_dict = year
        self.genres_dict = genre
        self.director_dict = director

        self.n_user = self.rating_df['user'].nunique() 
        self.n_item = self.rating_df['item'].nunique()
        self.n_year = len(set(self.year_dict.values()))
        self.n_genre = len(set(self.genres_dict.values()))
        self.n_director = len(set(self.director_dict.values()))

        self.users = self.rating_df['user'].unique() # unique 유저
        self.rating_df = rating.groupby('user')['item'].agg(lambda x : list(x)) # 유저별 본 아이템 리스트
        print("data load done")

    def get_num_context(self):
        """
        n_user,n_item,n_year,n_genre,n_director
        """
        return self.n_user, self.n_item, self.n_year, self.n_genre, self.n_director

    def pos_or_neg(self):
        return np.random.rand()

    def __len__(self):
        # batch는 유저별로
        return self.n_user
        
    def __getitem__(self,user_id):

        # 아이템 선택
        if self.pos_or_neg() > 0.5: # positive
            if self.mode == 'seq': 
                pos_set = self.rating_df[user_id][-self.valid_size:] # 뒤에꺼
            elif self.mode == 'static': # static
                if self.valid_size >= len(self.rating_df[user_id]):
                    pos_set = self.rating_df[user_id][:-10]
                else : pos_set = self.rating_df[user_id][:-self.valid_size]  # 앞에꺼
            self.item_selected = np.random.choice(pos_set)
            self.target = np.array(1)

        else: # negative
            pos_set = self.rating_df[user_id]
            self.item_selected = np.random.randint(1, self.n_item - 1)
            while self.item_selected in pos_set:
                self.item_selected = np.random.randint(1, self.n_item - 1)
            self.target = np.array(0)
            
        # 선택된 아이템과 context 정보 join
        genre_selected = self.genres_dict[self.item_selected]
        year_selected = self.year_dict[self.item_selected]
        director_selected = self.genres_dict[self.item_selected]

        if self.train_all:
            self.input = np.array([user_id, self.item_selected, year_selected, genre_selected, director_selected])
        else:
            self.input = np.array([user_id, year_selected, genre_selected,director_selected]) 
        return self.input, self.target