import random
import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import neg_sample, item_encoding


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

class BERT4RecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length
        self.num_item = args.item_size - 2
        self.mask_p = args.mask_p

    def __len__(self):
        return len(self.user_seq)

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
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            answer = [items[-2]]

        elif self.data_type == "test":
            input_ids = items[:-1]
            answer = [items[-1]]
        else:
            input_ids = items[:]
            answer = []

        tokens = []
        labels = []
        
        for ids in input_ids:
            prob = np.random.random()
            if prob < self.mask_p:
                prob /= self.mask_p

                # BERT 학습
                if prob < 0.8:
                    # masking
                    tokens.append(self.num_item + 1) # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                elif prob < 0.9:
                    tokens.append(np.random.randint(1, self.num_item + 1))  # item random sampling
                else:
                    tokens.append(ids)
                labels.append(ids)
                
            else:
                tokens.append(ids)
                labels.append(0)

        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        assert len(tokens) == self.max_len
        assert len(labels) == self.max_len

        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_tensors

class AutoRecDataset(Dataset):
    def __init__(self, args, inter_mat, answers_mat):
        self.args = args
        self.inter_mat = inter_mat
        self.answers = answers_mat.argsort(axis = 1)

        # valid data의 최소 길이
        # self.answers_minlen = min([len(answer) for answer in self.answers.values()])

    def __len__(self):
        return len(self.inter_mat)

    def __getitem__(self, index):
        user_id = index
        inter_mat = self.inter_mat[user_id]
        answers = self.answers[user_id][-10:]
       
        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(inter_mat, dtype=torch.float),
            torch.tensor(answers, dtype=torch.long),
        )

        return cur_tensors

class NCFDataset(Dataset):
    def __init__(self, df, answers_mat):
        # class init
        # self.args = args
        self.df = df
        # self.sampling_method = self.args.neg_sampling_method
        # self.n_negs = self.args.n_negs
        # self.neg_sample_num = self.args.neg_sample_num
        self.answers = answers_mat.argsort(axis = 1)

        # data split
        # self.user_list = self.df['user_idx'].unique()
        # self.new_df = pd.DataFrame()

        # for user_id in tqdm(self.user_list):
        #     self.temp_df = self.df[self.df['user_idx'] == user_id]

        #     if mode == 'submission':
        #         size = self.args.sub_per_user
        #         self.temp_df = self.temp_df[self.temp_df['label'] == 0].sample(size, replace = False, random_state=42)
        #         self.new_df = pd.concat([self.new_df, self.temp_df], ignore_index = True)

        #     else:
        #         self.positive_sample = self.temp_df[self.temp_df['label'] == 1]
                
        #         if mode == 'train':
        #             self.positive_sample = self.positive_sample[self.positive_sample['item_idx'].apply(lambda x : x in self.answers[user_id][:-10])] # negative sampling에서 valid 제외
                    
        #         elif mode == 'valid':
        #             neg_set_size = self.args.valid_per_user - 10
        #             self.positive_sample = self.positive_sample[self.positive_sample['item_idx'].apply(lambda x : x in self.answers[user_id][-10:])]

        #         if args.neg_sampling:
        #             if args.neg_sampling_method == 'n_neg':
        #                 neg_set_size = len(self.positive_sample) * self.n_negs 
        #             else:
        #                 neg_set_size = self.neg_sample_num

        #             self.negative_sample = self.temp_df[self.temp_df['label'] == 0].sample(neg_set_size, replace = False, random_state=42)
                
        #             self.new_df = pd.concat([self.new_df, self.positive_sample, self.negative_sample], ignore_index = True)
                
        #         else: 
        #             self.new_df = pd.concat([self.new_df, self.positive_sample], ignore_index = True)

        # self.df = self.new_df

        # dataloader getitem
        self.users = torch.tensor(self.df['user_idx'].values)
        self.items = torch.tensor(self.df['item_idx'].values)
        self.labels = torch.tensor(self.df['label'].values)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        users = self.users[index]
        items = self.items[index]
        labels = self.labels[index]
        answers = torch.tensor(self.answers[users][-10:])
        return (users, items, labels.float(), answers)
