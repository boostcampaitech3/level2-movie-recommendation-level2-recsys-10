import random
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sp

import torch
from torch.utils.data import Dataset

from utils import neg_sample

import os
from time import time

class PretrainDataset(Dataset):
    def __init__(self, args, user_seq, long_sequence):
        self.args = args
        self.user_seq = user_seq
        self.long_sequence = long_sequence
        self.max_len = args.max_seq_length
        self.part_sequence = []
        self.split_sequence()

    def split_sequence(self):
        for seq in self.user_seq:
            input_ids = seq[-(self.max_len + 2) : -2]  # keeping same as train set
            for i in range(len(input_ids)):
                self.part_sequence.append(input_ids[: i + 1])

    def __len__(self):
        return len(self.part_sequence)

    def __getitem__(self, index):

        sequence = self.part_sequence[index]  # pos_items
        # sample neg item for every masked item
        masked_item_sequence = []
        neg_items = []
        # Masked Item Prediction
        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.args.mask_p:
                masked_item_sequence.append(self.args.mask_id)
                neg_items.append(neg_sample(item_set, self.args.item_size))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)

        # add mask at the last position
        masked_item_sequence.append(self.args.mask_id)
        neg_items.append(neg_sample(item_set, self.args.item_size))

        # Segment Prediction
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            pos_segment = sequence[start_id : start_id + sample_length]
            neg_segment = self.long_sequence[
                neg_start_id : neg_start_id + sample_length
            ]
            masked_segment_sequence = (
                sequence[:start_id]
                + [self.args.mask_id] * sample_length
                + sequence[start_id + sample_length :]
            )
            pos_segment = (
                [self.args.mask_id] * start_id
                + pos_segment
                + [self.args.mask_id] * (len(sequence) - (start_id + sample_length))
            )
            neg_segment = (
                [self.args.mask_id] * start_id
                + neg_segment
                + [self.args.mask_id] * (len(sequence) - (start_id + sample_length))
            )

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # padding sequence
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + sequence
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0] * pad_len + masked_segment_sequence
        pos_segment = [0] * pad_len + pos_segment
        neg_segment = [0] * pad_len + neg_segment

        masked_item_sequence = masked_item_sequence[-self.max_len :]
        pos_items = pos_items[-self.max_len :]
        neg_items = neg_items[-self.max_len :]

        masked_segment_sequence = masked_segment_sequence[-self.max_len :]
        pos_segment = pos_segment[-self.max_len :]
        neg_segment = neg_segment[-self.max_len :]

        # Associated Attribute Prediction
        # Masked Attribute Prediction
        attributes = []
        for item in pos_items:
            attribute = [0] * self.args.attribute_size
            try:
                now_attribute = self.args.item2attribute[str(item)]
                for a in now_attribute:
                    attribute[a] = 1
            except:
                pass
            attributes.append(attribute)

        assert len(attributes) == self.max_len
        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len

        cur_tensors = (
            torch.tensor(attributes, dtype=torch.long),
            torch.tensor(masked_item_sequence, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
            torch.tensor(masked_segment_sequence, dtype=torch.long),
            torch.tensor(pos_segment, dtype=torch.long),
            torch.tensor(neg_segment, dtype=torch.long),
        )
        return cur_tensors


class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test", "submission"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]

        # submission [0, 1, 2, 3, 4, 5, 6]
        # answer None

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        elif self.data_type == "test":
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]
        else:
            input_ids = items[:]
            target_pos = items[:]  # will not be used
            answer = []

        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)

class MultiVAEDataset(Dataset):
    def __init__(self, path = '../data/'):
        self.path = path # default: '../data/'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.exist_users = []

        # data_path는 사용자의 디렉토리에 맞게 설정해야 합니다.
        data_path = os.path.join(self.path, 'train/train_ratings.csv')
        df = pd.read_csv(data_path)

        item_ids = df['item'].unique() # 아이템 고유 번호 리스트
        user_ids = df['user'].unique() # 유저 고유 번호 리스트
        self.n_items, self.n_users = len(item_ids), len(user_ids)
        
        # user, item indexing
        item2idx = pd.Series(data=np.arange(len(item_ids)), index=item_ids) 
        user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) 

        # dataframe indexing
        df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
        df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
        df.sort_values(['user_idx', 'time'], inplace=True)
        del df['item'], df['user']

        self.exist_items = list(df['item_idx'].unique())
        self.exist_users = list(df['user_idx'].unique())

        t1 = time()
        self.train_items, self.valid_items = {}, {}
        
        items = df.groupby("user_idx")["item_idx"].apply(list)
        
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

        print('Train/Vaild Split Complete. Takes in', time() - t1, 'sec')
        
        rows, cols = self.train_data['user'], self.train_data['item']
        self.train_input_data = sp.csr_matrix(
            (np.ones_like(rows), (rows, cols)), 
            dtype='float32',
            shape=(self.n_users, self.n_items))
        self.train_input_data = self.train_input_data.toarray()

        # bm25_weight
        # self.train_input_data = bm25_weight(self.train_input_data, K1=100, B=0.9)
        # values = self.train_input_data.data
        # indices = np.vstack((self.train_input_data.row, self.train_input_data.col))

        # i = torch.LongTensor(indices)
        # v = torch.FloatTensor(values)
        # shape = self.train_input_data.shape

        # self.train_input_data = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        return self.train_input_data[idx,:]

class MultiVAEValidDataset(Dataset):
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