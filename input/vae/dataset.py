
import os
from time import time
import scipy.sparse as sp
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, path = '../data/', mode = 'train'):
        self.path = path # default: '../data/'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.exist_users = []

        # data_path는 사용자의 디렉토리에 맞게 설정해야 합니다.
        data_path = os.path.join(self.path, 'train/train_ratings.csv')
        df = pd.read_csv(data_path)
        self.ratings_df = df.copy() # for submission
        self.n_train = len(df)

        item_ids = df['item'].unique() # 아이템 고유 번호 리스트
        user_ids = df['user'].unique() # 유저 고유 번호 리스트
        self.n_items, self.n_users = len(item_ids), len(user_ids)
        
        # user, item indexing
        # item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids) # item re-indexing (1~num_item) ; 아이템을 1부터 설정하는이유? 0을 아무것도 아닌 것으로 blank 하기 위해서
        item2idx = pd.Series(data=np.arange(len(item_ids)), index=item_ids) # item re-indexing (0~num_item-1) ; 아이템을 1부터 설정하는이유? 0을 아무것도 아닌 것으로 blank 하기 위해서
        user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) # user re-indexing (0~num_user-1)

        # dataframe indexing
        df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
        df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
        df.sort_values(['user_idx', 'time'], inplace=True)
        del df['item'], df['user']

        self.exist_items = list(df['item_idx'].unique())
        self.exist_users = list(df['user_idx'].unique())

        t1 = time()
        self.train_items, self.valid_items = {}, {}
        
        items = df.groupby("user_idx")["item_idx"].apply(list) # 유저 아이디 상관 없이, 순서대로 
        if mode == 'train':
            print('Creating interaction Train/ Vaild Split...')
            for uid, item in enumerate(items):            
                num_u_valid_items = min(int(len(item)*0.125), 10) # 유저가 소비한 아이템의 12.5%, 그리고 최대 10개의 데이터셋을 무작위로 Validation Set으로 활용한다.
                u_valid_items = np.random.choice(item, size=num_u_valid_items, replace=False)
                self.valid_items[uid] = u_valid_items
                # if len(item) <= 1000 and len(item) >= 30 : # 훈련에 사용할 데이터는 이상치 제거하고
                self.train_items[uid] = list(set(item) - set(u_valid_items))


            self.train_data = pd.concat({k: pd.Series(v) for k, v in self.train_items.items()}).reset_index(0)
            self.train_data.columns = ['user', 'item']

            self.valid_data = pd.concat({k: pd.Series(v) for k, v in self.valid_items.items()}).reset_index(0)
            self.valid_data.columns = ['user', 'item']
        
        if mode == 'train_all': #else
            print('Preparing interaction all train set')
            # for uid, item in enumerate(items):            
            #     self.train_items[uid] = item

            # self.train_data = pd.concat({k: pd.Series(v) for k, v in train_items.items()})
            # self.train_data.reset_index(0, inplace=True)
            # self.train_data.columns = ['user', 'item']
            self.train_data = pd.DataFrame()
            self.train_data['user'] = df['user_idx']
            self.train_data['item'] = df['item_idx']

        print('Train/Vaild Split Complete. Takes in', time() - t1, 'sec')
        
        rows, cols = self.train_data['user'], self.train_data['item']
        self.train_input_data = sp.csr_matrix(
            (np.ones_like(rows), (rows, cols)), 
            dtype='float32',
            shape=(self.n_users, self.n_items))
        self.train_input_data = self.train_input_data.toarray()

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        return self.train_input_data[idx,:]

class ValidDataset(Dataset):
    def __init__(self, train_dataset):
        self.n_users = train_dataset.n_users
        self.n_items = train_dataset.n_items
        self.train_input_data = train_dataset.train_input_data

    
        self.valid_data = train_dataset.valid_data
        rows, cols = self.valid_data['user'], self.valid_data['item']
        self.valid_input_data = sp.csr_matrix(
            (np.ones_like(rows), (rows, cols)), 
            dtype='float32',
            shape=(self.n_users, self.n_items))

        self.valid_input_data = self.valid_input_data.toarray()
    
    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        return self.train_input_data[idx, :], self.valid_input_data[idx,:]
