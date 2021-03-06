import random
import os
import time
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn import datasets

import torch
from torch.utils.data import Dataset
from utils import neg_sample
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

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
        # user ?????? ?????? 
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items # ???????????? ??????, sample??? ?????? ???????????? ???????????? ??????
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
        else: # for submission
            input_ids = items[:]
            target_pos = items[:]  # will not be used
            answer = []

        # negative sampling
        target_neg = []
        seq_set = set(items)
        for _ in input_ids: # seq_set??? ?????? items?????? target_neg??? input_ids??? ?????? ????????? ?????????
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids) # 50 - item ???
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        # input_ids ??? max_len ?????? ?????? ??????, (?????? ?????? ?????? 50??? ???????????? ?????????) max_len ?????? ???????????? ????????? ??????.
        input_ids = input_ids[-self.max_len :] 
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        # max_len ??? ?????? ????????? items ???????????? ????????? ????????? ??????
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
                torch.tensor(input_ids, dtype=torch.long), # user_id??? ?????? ?????? ??????
                torch.tensor(target_pos, dtype=torch.long), # ????????? ????????? pos
                torch.tensor(target_neg, dtype=torch.long), # ?????? ?????? ?????? ??????, ????????? ?????? ??????
                torch.tensor(answer, dtype=torch.long), # valid, test ?????? ?????????, train????????? target_pos??? ??????
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)

class WDDataset(Dataset):
    def __init__(self, args, data_type="train"):

        self.args = args
        #self.path = path
        self.data_type = data_type
        # data Path
        df = pd.read_csv(self.args.data_file)        
        #df = pd.read_csv(os.path.join('../data/', 'train/train_ratings.csv'))
        
        # for submission
        #self.rating_df = df.copy()
        #self.train_df = len(df)

        # get unique user and tiems
        self.user_ids = df['user'].unique()
        self.item_ids = df['item'].unique()
        
        # get number of users and items
        self.n_users, self.n_items = len(self.user_ids), len(self.item_ids)
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        # user, item indexing
        # item2idx : {data : 0~item_num , "index" : ?????? ??????}
        #self.item2idx = {"data" : np.arange(len(self.item_ids)), "index" : self.item_ids}
        self.item2idx = pd.Series(data=np.arange(self.n_items), index=self.item_ids) # item re-indexing
        # user2idx : {data }
        self.user2idx = pd.Series(data=np.arange(self.n_users), index=self.user_ids) # user re-indexing (0~num_user-1)

        # dataframe indexing
        df = pd.merge(df, pd.DataFrame({'item': self.item_ids, 'item_idx': self.item2idx[self.item_ids].values}), on='item', how='inner')
        df = pd.merge(df, pd.DataFrame({'user': self.user_ids, 'user_idx': self.user2idx[self.user_ids].values}), on='user', how='inner')
        df.sort_values(['user_idx', 'time'], inplace=True)
        del df['item'], df['user'], df['time']
        
        # user??? index ?????? ??????
        self.exist_users = list(df['user_idx'].unique())

        #genre_data to mulit-hot encoding
        #genre_df = pd.read_csv("/opt/ml/worksapce/level2-movie-recommendation-level2-recsys-10/input/data/train/genres.tsv", sep='\t')
        genre_df = pd.read_csv(os.path.join('../data/', 'train/genres.tsv'), sep='\t')

        # genre_mulit = DataFrame(genres, item_idx)
        self.genre_mulit = self.genre_items_mulithot(genre_df)

        # train, valid split
        #self.trainsets, self.trainsets_unique, self.validsets, self.valid_unique = self.train_valid_split(df)

        if self.data_type == "train":
            self.datasets ,self.datasets_unique , _, _ = self.train_valid_split(df)
            self.mask_prob = 0.7
        elif self.data_type == "valid":
            _, _, self.datasets ,self.datasets_unique  = self.train_valid_split(df)
            self.mask_prob = 0.7


    def genre_items_mulithot(self, genre_data):
        # gnre mulit-hot encoding
        genre_dict = {genre:i for i, genre in enumerate(set(genre_data['genre']))}
        genre_data['genre']  = genre_data['genre'].map(lambda x : genre_dict[x])
        sum_genre = list()
        for item in self.item_ids:
            sum_genre.append([item, genre_data[genre_data['item']==item]['genre'].values])
        sum_genre = pd.DataFrame(sum_genre , columns=['item', 'genre'])
        
        # Mulit-Labeling
        mlb = MultiLabelBinarizer()
        genre_label = mlb.fit_transform(sum_genre['genre'])
        sum_genre = pd.concat([sum_genre['item'],pd.DataFrame(genre_label, columns=genre_dict)], axis = 1)
        sum_genre = pd.merge(sum_genre, pd.DataFrame({'item': self.item_ids, 'item_idx': self.item2idx[self.item_ids].values}), on='item', how='inner')
        sum_genre.sort_values(['item_idx'], inplace=True)
        del sum_genre['item']
        
        return sum_genre 

    def train_valid_split(self, df):
        items = df.groupby("user_idx")["item_idx"].apply(list)
        # {"user_id" : [items]}
        train_set, valid_set = {} , {}
        train_unique, valid_unique = [] , []
        print("----train_valid set split by user_idx----")
        for uid, item in tqdm(enumerate(items)):
            # ????????? ????????? item??? 12.5% ?????? ?????? 10 ?????? valid_set ????????? ??????
            num_u_valid_set = min(int(len(item)*0.125), 10)
            u_valid_set = np.random.choice(item, size=num_u_valid_set, replace=False)
            
            train_set[uid] = list(set(item) - set(u_valid_set))
            train_unique.extend(train_set[uid])
            valid_set[uid] = u_valid_set
            valid_unique.extend(valid_set[uid])

        
        return train_set , set(train_unique), valid_set , set(valid_unique)
        

    # def make_pos_next_items(self, user):
        # ?????? user??? ???????????? positive, negative ??????
        #mask_prob = self.args.mask_prob
        # seq = self.datasets[user]
        # datasets = set(sum(self.datasets.values(),[]))
        # tokens = []
        # target = []

        # for s in seq: # user??? ?????? items sets??? ????????? ??????     
        #     negative_items = np.random.choice(list(datasets - set(seq)), self.args.num_negative, replace=False)
        #     token_in = [user, s] # [user, item]
        #     item_genres = self.genre_mulit[self.genre_mulit["item_idx"]==s].iloc[:,1:].values.tolist()
        #     seq.remove(s)
        #     if len(seq) != 1: # seq??? ??????????????? pass
        #         for q in seq: # s??? ????????? items?????? positive ?????? ?????????
        #             token_in.extend([q])
        #             # item genres, next_genres
        #             token_in.extend(*item_genres) 
        #             token_in.extend(*(self.genre_mulit[self.genre_mulit["item_idx"]==q].iloc[:,1:].values.tolost()))
        #             tokens.append(token_in)

        #             target.append([1])
        #             token_in = [user, s]

        #     for negative in negative_items: # negative ?????? ??????
        #         token_in.extend([negative]) 
        #         token_in.extend(*item_genres)
        #         token_in.extend(*(self.genre_mulit[self.genre_mulit["item_idx"]==negative].iloc[:,1:].values.tolost()))
                
        #         target.append([0])
        #         tokens.append(token_in)
        #         token_in = [user, s]

        # [[user, item, next, item_genres, next_genres]] , [[target]]
        # return tokens , target

    def random_pos_next_items(self, user):
    # ?????? user??? ???????????? ?????? ????????? positive, negative ???????????? ?????????
        mask_prob = self.mask_prob
        item = np.random.choice(self.datasets[user], 1, replace=False)
        datasets_unique = self.datasets_unique
        target = []
        
        #negative_items = np.random.choice(list(datasets - set(item)), self.args.num_negative, replace=False)
        token = [user, item[0]] # [user, item]
        item_genres = self.genre_mulit[self.genre_mulit["item_idx"]==item[0]].iloc[:,0:-1].values.tolist()
        token.extend(*item_genres)
        prob = np.random.random()
        if prob < mask_prob:
            # positive case
            positive_next = np.random.choice(list(datasets_unique - set(item)), 1, replace=False)
            token.append(positive_next[0])
            token.extend(self.genre_mulit[self.genre_mulit["item_idx"]==positive_next[0]].iloc[:,0:-1].values[0])
            target.append(1)

        else:
            # negative case
            negative_next = np.random.choice(list(datasets_unique - set(item)), 1, replace=False)
            token.append(negative_next[0])
            token.extend(self.genre_mulit[self.genre_mulit["item_idx"]==negative_next[0]].iloc[:,0:-1].values[0])
            target.append(0)


        # [user, item, item_genres, next, next_genres] 1 + 1 + 18 + 1 + 18 = 39 , [target]
        return token , target

    def __getitem__(self, idx): # idx = batch_size 
        # train, valid set split
        user = self.exist_users[idx]
        X , y = self.random_pos_next_items(user)
        return torch.tensor(X), torch.tensor(y)
        #return X, y

    def __len__(self):
        # user??? ??? ??????
        return self.n_users

