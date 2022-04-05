import random

import torch
from torch.utils.data import Dataset

from utils import neg_sample


class DeepFMDataset(Dataset):
    def __init__(self, deepfm_data_dir):
        # deepfm_train.json 불러와서 input, targer tensor 변환
        self.df = pd.read_json(deepfm_data_dir)
        self.user_col = torch.tensor(self.df.loc[:, 'user'])
        self.item_col = torch.tensor(self.df.loc[:, 'item'])
        self.year_col = torch.tensor(self.df.loc[:, 'year'])
        self.genre_col = torch.tensor(self.df.loc[:, 'genre'])
        
        self.input_tensor = torch.cat([self.user_col.unsqueeze(1), self.item_col.unsqueeze(1), 
                                        self.year_col.unsqueeze(1), self.genre_col], dim=1).long()
        self.target_tensor = torch.tensor(list(self.df.loc[:, 'rating'])).long()
        
    def __len__(self,index):
        return self.target_tensor.size(0)
        
    def __getitem__(self,index):
        return self.input_tensor[index], self.target_tensor[index]

    def get_num_context(self):
        users = list(set(self.user_col))
        items = list(set(self.item_col))
        genres = list(set(self.year_col))

        n_user = len(users)
        n_item = len(items)
        n_genre = len(genres)

        return n_user,n_item,n_genre