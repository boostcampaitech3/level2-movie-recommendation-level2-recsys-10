import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy import sparse
from time import time

from datasets import MultiVAEDataset, MultiVAEValidDataset
from torch.utils.data import DataLoader
from models import MultiVAE
from utils import Recall_at_k_batch
from preprocessing import numerize

import wandb

def submission_multi_vae(args, model):
    DATA_DIR = args.data
    rating_df = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)

    test_users = rating_df["user"].unique()
    test_item_ids = rating_df['item'].unique()

    test_unique_uid = pd.unique(rating_df['user'])
    test_unique_sid = pd.unique(rating_df['item'])

    n_items = len(pd.unique(rating_df['item']))

    show2id = dict((sid, i) for (i, sid) in enumerate(test_unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(test_unique_uid))

    test_rating_df = numerize(rating_df, profile2id, show2id)

    n_users = test_rating_df['uid'].max() + 1
    rows, cols = test_rating_df['uid'], test_rating_df['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                            (rows, cols)), dtype='float64',
                            shape=(n_users, n_items))

    test_data_tensor = torch.FloatTensor(data.toarray()).to(device)

    recon_batch, mu, logvar = model(test_data_tensor)

    id2show = dict(zip(show2id.values(),show2id.keys()))
    id2profile = dict(zip(profile2id.values(),profile2id.keys()))

    result = []

    for user in range(len(recon_batch)):
        rating_pred = recon_batch[user]
        rating_pred[test_data_tensor[user].reshape(-1) > 0] = 0

        idx = np.argsort(rating_pred.detach().cpu().numpy())[-10:][::-1]
        for i in idx:
            result.append((id2profile[user], id2show[i]))

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        "output/" + args.model_name + ".csv", index=False
    )

def main():
    ## 각종 파라미터 세팅
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')

    parser.add_argument('--data', type=str, default='../data/train/',
                        help='Movielens dataset location')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    parser.add_argument("--model_name", default="MultiVAE", type=str)
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='Multi_VAE.pt',
                        help='path to save the final model')
    parser.add_argument("--wandb_name")

    args = parser.parse_args()

    # Set the random seed manually for reproductibility.
    torch.manual_seed(args.seed)

    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        args.cuda = True

    global device
    device = torch.device("cuda" if args.cuda else "cpu")

    # wandb init
    wandb.init(project="movierec_train", entity="egsbj")
    wandb.run.name = args.wandb_name

    if args.model_name == 'MultiVAE':
        train_batch_size = 500
        valid_batch_size = 1000

        # 만들어준 데이터 셋을 바탕으로 Dataset과 Dataloader를 정의
        train_dataset = MultiVAEDataset()
        valid_dataset = MultiVAEValidDataset(train_dataset = train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, pin_memory=True, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=False, pin_memory=True, shuffle=False)

        # 모델 정의
        n_items = train_dataset.n_items
        model = MultiVAE(args, p_dims=[200, 600, n_items]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

        update_count = 0
        best_r10 = -np.inf

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time()
            ###### train ######
            model.train()
            train_loss = 0.0
            start_time = time()

            for batch_idx, batch_data in enumerate(train_loader):
                input_data = batch_data.to(device)
                optimizer.zero_grad()

                if args.total_anneal_steps > 0:
                    anneal = min(args.anneal_cap, 
                                    1. * update_count / args.total_anneal_steps)
                else:
                    anneal = args.anneal_cap

                recon_batch, mu, logvar = model(input_data)
                
                loss = model.loss_function(recon_batch, input_data, mu, logvar, anneal)
                
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                update_count += 1        

                log_interval = 100
                if batch_idx % log_interval == 0 and batch_idx > 0:
                    elapsed = time() - start_time
                    print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                            'loss {:4.2f}'.format(
                                epoch, batch_idx, len(range(0, 6807, batch_size)),
                                elapsed * 1000 / log_interval,
                                train_loss / log_interval))

                    start_time = time()
                    train_loss = 0.0

            ###### eval ######
            recall10_list = []
            recall20_list = []
            total_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch_data in valid_loader:
                    input_data, label_data = batch_data # label_data = validation set 추론에도 사용되지 않고 오로지 평가의 정답지로 사용된다. 
                    input_data = input_data.to(device)
                    label_data = label_data.to(device)
                    label_data = label_data.cpu().numpy()
                    
                    if args.total_anneal_steps > 0:
                        anneal = min(args.anneal_cap, 
                                    1. * update_count / args.total_anneal_steps)
                    else:
                        anneal = args.anneal_cap

                    recon_batch, mu, logvar = model(input_data)

                    loss = model.loss_function(recon_batch, input_data, mu, logvar, anneal)

                    total_loss += loss.item()
                    recon_batch = recon_batch.cpu().numpy()
                    recon_batch[input_data.cpu().numpy().nonzero()] = -np.inf

                    recall10 = Recall_at_k_batch(recon_batch, label_data, 10)
                    recall20 = Recall_at_k_batch(recon_batch, label_data, 20)
                    
                    recall10_list.append(recall10)
                    recall20_list.append(recall20)
            
            total_loss /= len(range(0, 6807, 1000))
            r10_list = np.concatenate(recall10_list)
            r20_list = np.concatenate(recall20_list)
                    
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                    'r10 {:5.3f} | r20 {:5.3f}'.format(
                        epoch, time() - epoch_start_time, total_loss, np.mean(r10_list), np.mean(r20_list)))
            print('-' * 89)
            
            wandb.log({"valid loss" : total_loss,
            "r20" : np.mean(r20_list), 
            "r10" : np.mean(r10_list)})

            if np.mean(r10_list) > best_r10:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_r10 = np.mean(r10_list)

    # inference
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    submission_multi_vae(args, model)

if __name__ == "__main__":
    main()