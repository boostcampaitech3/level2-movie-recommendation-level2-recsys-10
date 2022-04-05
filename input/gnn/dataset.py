import random as rd
import pandas as pd
import numpy as np
import scipy.sparse as sp
from collections import Counter
import torch
from time import time
import pandas as pd
import os
from datetime import datetime
import easydict
import tqdm
import scipy.sparse as sp
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from utils import get_probability_from_arr

class BaseDataset(Dataset):
    def __init__(self, path, mode = 'train'):
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

        self.item_counter = Counter(df['item_idx'])
        
        # create interactions/ratings matrix 'R' # dok = dictionary of keys
        print('Creating interaction matrices R_train and R_test...')
        t1 = time()
        self.R_train = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32) 
        self.R_test = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.valid_set = {}, {}
        
        items = df.groupby("user_idx")["item_idx"].apply(list) # 유저 아이디 상관 없이, 순서대로 
        for uid, item in tqdm.tqdm(enumerate(items)):
            # for i in item:
            #     self.R_train[uid, i] = 1.        
            # self.train_items[uid] = item
            
            if mode == 'train':
                num_u_valid_set = min(int(len(item)*0.125), 10) # 유저가 소비한 아이템의 12.5%, 그리고 최대 10개의 데이터셋을 무작위로 Validation Set으로 활용한다.
                u_valid_set = np.random.choice(item, size=num_u_valid_set, replace=False)
                for i in set(item) - set(u_valid_set):
                    self.R_train[uid, i] = 1.
                self.train_items[uid] = list(set(item) - set(u_valid_set))

                for i in u_valid_set:
                    self.R_test[uid, i] = 1.
                self.valid_set[uid] = u_valid_set
        
            if mode == 'train_all':
                for i in item:
                    self.R_train[uid, i] = 1.
                self.train_items[uid] = item

        print('Complete. Interaction matrices R_train and R_test created in', time() - t1, 'sec')
        
        # FIXME; Take too much time to return popular negative sampling
        # for Popular negative Sampling | Method(1): prob. = freq.
        # self.train_unseen_items = [] # index: user_id, value: (list)- user unseen items
        # self.train_unseen_items_probs = [] # index: user_id, value: (list)- user unseen items' probabilities (<-frequencies)
        # for user in self.exist_users:
        #     train_unseen_items = list(set(self.exist_items) - set(self.train_items[user])) # (list)- user unseen items
        #     self.train_unseen_items.append(train_unseen_items)
        #     train_unseen_items_counts = [self.item_counter[item] for item in train_unseen_items]
        #     self.train_unseen_items_probs.append(get_probability_from_arr(train_unseen_items_counts))


    def __len__(self):
        return self.n_users


    def sample_pos_items_for_u(self, u, num):
        pos_items = self.train_items[u]
        n_pos_items = len(pos_items)
        pos_batch = []
        while True:
            if len(pos_batch) == num: break
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]

            if pos_i_id not in pos_batch:
                pos_batch.append(pos_i_id)
        return pos_batch

    def sample_neg_items_for_u(self, u, num):
        neg_items = []
        while True:
            if len(neg_items) == num: break
            neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
            if neg_id not in self.train_items[u] and neg_id not in neg_items:
                neg_items.append(neg_id)
        return neg_items

    # FIXME 이 부분을 동적으로 epoch 에 따른 값을 적용시키게 만드려면?
    def sample_popular_neg_items_for_u(self, u, num):
        neg_items = []
        while True:
            if len(neg_items) == num: break
            neg_id = np.random.choice(self.train_unseen_items[u], size = 1, replace = False, p=self.train_unseen_items_probs[u])[0]
            if neg_id not in neg_items:
                neg_items.append(neg_id)
        return neg_items

    def __getitem__(self, idx):
        # TODO batch size 보다 user의 수가 더 적을 경우??? 일단 이런 경우는 무시하고 진행
        user = self.exist_users[idx]
        pos_item = self.sample_pos_items_for_u(user, 1)[0]
        neg_item = self.sample_neg_items_for_u(user, 1)[0] #if np.random.uniform(0,1,1) < 0.5 else self.sample_popular_neg_items_for_u(user, 1)[0]
        return user, pos_item, neg_item

    # if exist, get adjacency matrix
    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(os.path.join(self.path, 's_adj_mat.npz'))
            print('Loaded adjacency-matrix (shape:', adj_mat.shape,') in', time() - t1, 'sec.')

        except Exception:
            print('Creating adjacency-matrix...')
            adj_mat = self.create_adj_mat()
            sp.save_npz(os.path.join(self.path, 's_adj_mat.npz'), adj_mat)
        return adj_mat
    
    # create adjancency matrix
    def create_adj_mat(self):
        t1 = time()
        
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R_train.tolil() # to list of lists

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('Complete. Adjacency-matrix created in', adj_mat.shape, time() - t1, 'sec.')

        t2 = time()

        # normalized adjacency matrix
        def normalized_adj_single(adj):
            
            ### 논문 수식 (8) ###
            rowsum = np.array(adj.sum(1))                
            d_inv = np.power(rowsum, -.5).flatten()     
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)                 
            norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv) 
            
            return norm_adj.tocoo()

        print('Transforming adjacency-matrix to Normalized-adjacency matrix...')
        ngcf_adj_mat = normalized_adj_single(adj_mat)

        print('Complete. Transformed adjacency-matrix to Normalized-adjacency matrix in', time() - t2, 'sec.')
        return ngcf_adj_mat.tocsr()

    # create collections of N items that users never interacted with
    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)




    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path # default: '../data/'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.exist_users = []
        self.batch_size = batch_size

        # data_path는 사용자의 디렉토리에 맞게 설정해야 합니다.
        data_path = os.path.join(self.path, 'train/train_ratings.csv')
        df = pd.read_csv(data_path)
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

        self.exist_users = list(df['user_idx'].unique())

        # create interactions/ratings matrix 'R' # dok = dictionary of keys
        print('Creating interaction matrices R_train and R_test...')
        t1 = time()
        self.R_train = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32) 
        self.R_test = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.valid_set = {}, {}
        
        items = df.groupby("user_idx")["item_idx"].apply(list) # 유저 아이디 상관 없이, 순서대로 
        for uid, item in tqdm.tqdm(enumerate(items)):
            # for i in item:
            #     self.R_train[uid, i] = 1.        
            # self.train_items[uid] = item

            num_u_valid_set = min(int(len(item)*0.125), 10) # 유저가 소비한 아이템의 12.5%, 그리고 최대 10개의 데이터셋을 무작위로 Validation Set으로 활용한다.
            u_valid_set = np.random.choice(item, size=num_u_valid_set, replace=False)
            for i in set(item) - set(u_valid_set):
                self.R_train[uid, i] = 1.
            self.train_items[uid] = list(set(item) - set(u_valid_set))

            for i in u_valid_set:
                self.R_test[uid, i] = 1.
            self.valid_set[uid] = u_valid_set


        print('Complete. Interaction matrices R_train and R_test created in', time() - t1, 'sec')        


    # if exist, get adjacency matrix
    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(os.path.join(self.path, 's_adj_mat.npz'))
            print('Loaded adjacency-matrix (shape:', adj_mat.shape,') in', time() - t1, 'sec.')

        except Exception:
            print('Creating adjacency-matrix...')
            adj_mat = self.create_adj_mat()
            sp.save_npz(os.path.join(self.path, 's_adj_mat.npz'), adj_mat)
        return adj_mat
    
    # create adjancency matrix
    def create_adj_mat(self):
        t1 = time()
        
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R_train.tolil() # to list of lists

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('Complete. Adjacency-matrix created in', adj_mat.shape, time() - t1, 'sec.')

        t2 = time()

        # normalized adjacency matrix
        def normalized_adj_single(adj):
            
            ### 논문 수식 (8) ###
            rowsum = np.array(adj.sum(1))                
            d_inv = np.power(rowsum, -.5).flatten()     
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)                 
            norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv) 
            
            return norm_adj.tocoo()

        print('Transforming adjacency-matrix to Normalized-adjacency matrix...')
        ngcf_adj_mat = normalized_adj_single(adj_mat)

        print('Complete. Transformed adjacency-matrix to Normalized-adjacency matrix in', time() - t2, 'sec.')
        return ngcf_adj_mat.tocsr()

    # create collections of N items that users never interacted with
    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    # sample data for mini-batches
    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


