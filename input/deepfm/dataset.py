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
                pos_set = self.rating_df[user_id][-self.valid_size:]
            elif self.mode == 'static': # static
                if self.valid_size >= len(self.rating_df[user_id]):
                    pos_set = self.rating_df[user_id][:-10]
                else : pos_set = self.rating_df[user_id][:-self.valid_size]  
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