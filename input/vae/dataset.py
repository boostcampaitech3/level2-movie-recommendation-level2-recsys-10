
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

        ############### item based outlier ###############
        # # 아이템 기준 outlier 제거 - 이용율 0.3% 미만인 아이템 제거 (영구히 제거)
        # item_freq_df = (df.groupby('item')['user'].count()/df.user.nunique()).reset_index()
        # item_freq_df.columns = ['item', 'item_freq']
        # # df = df.merge(item_freq_df, on='item').query('item_freq > 0.003')
        # # df = df.merge(item_freq_df, on='item').query('item_freq > 0.005')
        # df = df.merge(item_freq_df, on='item').query('item_freq > 0.01')
        # del df['item_freq'] # 소명을 다하고 삭제! 

        self.ratings_df = df.copy() # for submission
        self.n_train = len(df)

        item_ids = df['item'].unique() # 아이템 고유 번호 리스트
        user_ids = df['user'].unique() # 유저 고유 번호 리스트
        self.n_items, self.n_users = len(item_ids), len(user_ids)
        
        # user, item indexing
        # item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids) # item re-indexing (1~num_item) ; 아이템을 1부터 설정하는이유? 0을 아무것도 아닌 것으로 blank 하기 위해서
        self.item2idx = pd.Series(data=np.arange(len(item_ids)), index=item_ids) # item re-indexing (0~num_item-1) ; 아이템을 1부터 설정하는이유? 0을 아무것도 아닌 것으로 blank 하기 위해서
        self.user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) # user re-indexing (0~num_user-1)

        # dataframe indexing
        df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': self.item2idx[item_ids].values}), on='item', how='inner')
        df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': self.user2idx[user_ids].values}), on='user', how='inner')
        df.sort_values(['user_idx', 'time'], inplace=True)
        del df['item'], df['user']

        self.exist_items = list(df['item_idx'].unique())
        self.exist_users = list(df['user_idx'].unique())

        ############### Used by Sampler ###############
        # # 1. user-based outlier - 상위 20퍼센트 영화를 본 친구들 Weight=0 지정
        # self.user_weights = np.ones_like(self.exist_users)
        # outlier_users = df['user_idx'].unique()[df.groupby('user_idx').item_idx.count()/df['item_idx'].nunique() >= 0.4]
        # self.user_weights[outlier_users] = 0

        t1 = time()
        self.train_items, self.valid_items = {}, {}
        
        items = df.groupby("user_idx")["item_idx"].apply(list) # 유저 아이디 상관 없이, 순서대로 
        if mode == 'train':
            print('Creating interaction Train/ Vaild Split...')
            for uid, item in enumerate(items):            
                num_u_valid_items = min(int(len(item)*0.125), 10) # 유저가 소비한 아이템의 12.5%, 그리고 최대 10개의 데이터셋을 무작위로 Validation Set으로 활용한다.
                u_valid_items = np.random.choice(item, size=num_u_valid_items, replace=False)
                self.valid_items[uid] = u_valid_items
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

class BeforeNoiseUnderSamplingDataset(BaseDataset):
    def __init__(self, path='../data/', mode='train'):
        super().__init__(path, mode)

    # def noise_without_pos(self, u, num):
    #     pos_items = self.train_input_data[u]
    #     # n_pos_items = len(pos_items)
    #     pos_batch = []
    #     while True:
    #         if len(pos_batch) == num: break
    #         pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
    #         pos_i_id = pos_items[pos_id]

    #         if pos_i_id not in pos_batch:
    #             pos_batch.append(pos_i_id)
    #     return pos_batch


    def __getitem__(self, idx):
        # noise = np.random.choice(2, size=[*self.train_input_data.shape], p=[0.9, 0.1])
        # train_input_data_noised = self.train_input_data - noise
        # train_input_data_noised[train_input_data_noised < 0] = 0
        # return train_input_data_noised[idx,:]
        # noise = np.random.choice(2, size=[*self.train_input_data.shape], p=[0.9, 0.1])
        # noise = np.random.choice(2, size= len(self.train_input_data[idx,:]),  p=[0.9, 0.1]).astype(np.float32)
        # train_input_data_noised = self.train_input_data + noise
        # train_input_data_noised[train_input_data_noised < 0] = 0
        return self.train_input_data[idx,:], np.random.randint(0,2,size=self.train_input_data.shape[1]).astype(np.float32)