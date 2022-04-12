import os
from models import AdmmSlim
from datasets import BaseDataset, ValidDataset
import numpy as np
import pandas as pd

import wandb
import argparse

parser = argparse.ArgumentParser()

# env parameter
parser.add_argument('--seed', type=int, default='42', help='')
# parser.add_argument('--dataset', type=str)

parser.add_argument("--lambda_1", type=int, default='1', help=" ")
parser.add_argument("--lambda_2", type=int, default='500', help=" ")
parser.add_argument("--rho", type=int, default='10000', help=" ")
parser.add_argument("--n_iter", type=int, default='50', help=" ")
parser.add_argument("--eps_rel", type=float, default='1e-4', help=" ")
parser.add_argument("--eps_abs", type=float, default='1e-3', help=" ")
parser.add_argument("--verbose", type=bool, default=True, help=" ")

parser.add_argument("--data_dir", type=str, default='../data/', help=" ")
parser.add_argument("--output_dir", type=str, default='./output', help=" ")
args = parser.parse_args()

### import dataset
train_dataset = BaseDataset(path = os.path.join(args.data_dir)) # args.path = '../data/'
# valid_dataset = ValidDataset(train_dataset = train_dataset)

train_X = train_dataset.train_input_data
# valid_X = valid_dataset.valid_input_data

### import model
model = AdmmSlim(
    lambda_1 = args.lambda_1,
    lambda_2 = args.lambda_2,
    rho = args.rho,
    n_iter = args.n_iter,
    eps_rel = args.eps_rel,
    eps_abs = args.eps_abs,
    verbose = args.verbose
    )

### train
model.fit(train_X)

### output
y_predict = model.predict(train_X)
unseen_predict = y_predict*(1-train_X)
top_items = np.argsort(unseen_predict, axis=1)[:, -10:]

# back-labeling
idx2item = train_dataset.item2idx.reset_index(0)
idx2item.columns = ['item', 'item_idx']
idx2user = train_dataset.user2idx.reset_index(0)
idx2user.columns = ['user', 'user_idx']

temp = pd.concat({k: pd.Series(v) for k, v in enumerate(top_items)}).reset_index(0)
temp.columns = ['user_idx', 'item_idx']

temp = temp.merge(idx2user, on='user_idx')
temp = temp.merge(idx2item, on='item_idx')

del temp['user_idx'], temp['item_idx']

output = temp.sort_values('user')
output.index = range(len(output))

output.to_csv('submission_slim.csv', index=False)



