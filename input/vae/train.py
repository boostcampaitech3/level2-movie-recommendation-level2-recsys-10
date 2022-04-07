from importlib.resources import path
import torch, random
import numpy as np
from time import time
from datetime import datetime
from pytz import timezone

from copy import deepcopy
from utils import set_seed, get_lr, increment_path, check_path, early_stopping, Recall_at_k_batch, NDCG_binary_at_k_batch
from models import multivae, recvae
from importlib import import_module
import os
from torch.utils.data import DataLoader
from dataset import BaseDataset, ValidDataset

import wandb
import argparse

parser = argparse.ArgumentParser()

# env parameter
parser.add_argument('--seed', type=int, default='42', help=' ')
parser.add_argument('--dataset', type=str)

# model parameter
parser.add_argument('--model', type=str, default='RecVAE', help='model type (default: RecVAE)')
parser.add_argument('--hidden_dim', type=int, default=600)
parser.add_argument('--latent_dim', type=int, default=200)
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--gamma', type=float, default=0.005)
parser.add_argument('--dropout_rate', type=float, default=0.5)

# train parameter (with valid)
parser.add_argument("--epochs", type=int, default=3000, help="number of epochs")

parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-3)#5e-4
# parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--n-enc_epochs', type=int, default=3)
parser.add_argument('--n-dec_epochs', type=int, default=1)
parser.add_argument('--not-alternating', type=bool, default=False) # default = False, True
parser.add_argument("--eval_N", type=int, default=1, help=" ")
parser.add_argument("--k", type=int, default=10, help=" ")

# path
parser.add_argument("--data_dir", type=str, default='../data/', help=" ")
# parser.add_argument("--data_dir", type=str, default='/opt/ml/workspace/level2-movie-recommendation-level2-recsys-10/input/data', help=" ")

parser.add_argument("--output_dir", type=str, default='./output', help=" ")

# utilities
parser.add_argument("--wandb_name", type=str, default='-', help=" ")

args = parser.parse_args()

##### wandb init #####
# wandb.init(project="movierec_train", entity="egsbj")
# wandb.run.name = args.model + args.wandb_name
# wandb.config.update(args)

# -- seed
set_seed(args.seed)

# -- device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# -- save dir
save_dir_name = args.model
save_dir_path = increment_path(os.path.join(args.output_dir, save_dir_name))
check_path(save_dir_path)

# -- dataset
train_dataset = BaseDataset(path = os.path.join(args.data_dir)) # args.path = '../data/'
valid_dataset = ValidDataset(train_dataset = train_dataset)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False)

# -- model
model_module = getattr(import_module("models"), args.model)  # default: RecVAE
model = model_module(
    hidden_dim = args.hidden_dim, 
    latent_dim = args.latent_dim, 
    input_dim = train_dataset.n_items,
    dropout_rate = args.dropout_rate
).to(device)

# -- optimizer
decoder_params = set(model.decoder.parameters())
encoder_params = set(model.encoder.parameters())

optimizer_encoder = torch.optim.Adam(encoder_params, lr=args.lr)
optimizer_decoder = torch.optim.Adam(decoder_params, lr=args.lr)

# -- others; training env preset
cur_best_metric = 0
cur_best_loss, stopping_step, should_stop = 1e3, 0, False
today = datetime.now(timezone('Asia/Seoul'))
print("Start at " + str(today))
print("Using " + str(device) + " for computations")
print("Params on CUDA: " + str(next(model.parameters()).is_cuda))
results = {"Epoch": [], "Loss": [], "Valid_Loss":[], "Recall": [], "Training Time": []} #, "NDCG": []

# -- trainer module
def trainer(model, opts, train_loader, n_epochs, beta=None, gamma=1):
    temp_total_loss = 0.0
    model.train()
    for epoch in range(n_epochs):
        for batch_index, batch_data in enumerate(train_loader):
            input_data = batch_data.to(device)

            for optimizer in opts:
                optimizer.zero_grad()

            recon_batch, mu, logvar = model(input_data)
            _, loss = model.loss_function(recon_batch, input_data, mu, logvar, beta=beta, gamma=gamma)
            loss.backward()

            for optimizer in opts:
                optimizer.step()
            
            temp_total_loss += loss.item()

    return temp_total_loss

###### ###### ###### ###### ###### [ train & validation ] ###### ###### ###### ###### ###### 
best_score = -1
for epoch in range(args.epochs):
    # -- train
    running_loss = 0.0
    print()
    print("="*40, f"| [EPOCH: {epoch:3}/{args.epochs}] |", "="*40)
    t1 = time()

    if args.not_alternating:
        model.dropout_rate = 0.5
        loss = trainer(model, opts=[optimizer_encoder, optimizer_decoder], train_loader=train_loader, n_epochs=1)
        running_loss += loss
    else:
        model.dropout_rate = 0.5
        enc_loss = trainer(model, opts=[optimizer_encoder], train_loader=train_loader, n_epochs=args.n_enc_epochs)
        model.update_prior()
        model.dropout_rate = 0
        dec_loss = trainer(model, opts=[optimizer_decoder], train_loader=train_loader, n_epochs=args.n_dec_epochs)        

        running_loss += enc_loss+dec_loss # TODO 각각의 학습 정도를 확인하기위해 이정도는 print로 찍어줄까?
    
    current_lr = get_lr(optimizer_decoder)
    training_time = time()-t1
    print(f"[Train] time: {training_time:4.2}s | Loss: {running_loss:4.4} | lr: {current_lr}")

    # -- validation
    # print valid evaluation metrics every N epochs (provided by args.eval_N)
    if epoch % args.eval_N  == (args.eval_N - 1):
        recall_list = []
        ndcg_list = []
        total_loss = 0.0
        model.eval()
        with torch.no_grad():
            t2 = time()
            for batch_data in valid_loader:
                input_data, label_data = batch_data # label_data = validation set 추론에도 사용되지 않고 오로지 평가의 정답지로 사용된다. 
                input_data = input_data.to(device)
                label_data = label_data.cpu().numpy()#.to(device)

                recon_batch, mu, logvar = model(input_data)
                _, loss = model.loss_function(recon_batch, input_data, mu, logvar, gamma=1, beta=None)

                total_loss += loss.item()


                # calculate evaluation metrics
                recon_batch = recon_batch.cpu().numpy()
                recon_batch[input_data.cpu().numpy().nonzero()] = -np.inf

                _recall = Recall_at_k_batch(recon_batch, label_data, args.k) # (users:batch_size, recall:1)
                # _ndcg = NDCG_binary_at_k_batch(recon_batch, label_data, args.k) # TODO 현재 데이터 셋에 맞게 수정해준다. 

                recall_list.append(_recall)
                # ndcg_list.append(_ndcg)

        total_loss /= (train_dataset.n_users//args.batch_size) 
        recall_list = (np.concatenate(recall_list))
        recall = np.mean(recall_list)
        # ndcg = np.concatenate(ndcg_list).mean()
        print(f"[Valid] time: {time()-t2:4.2}s | Loss: {total_loss:4.4} | Recall@{args.k}: {recall:.4}")
        cur_best_metric, stopping_step, should_stop = early_stopping(recall, cur_best_metric, stopping_step, flag_step=20)

        # save results in dict
        results['Epoch'].append(epoch)
        results['Loss'].append(running_loss)
        results['Valid_Loss'].append(total_loss)
        results['Recall'].append(recall)
        # results['NDCG'].append(ndcg.item())
        results['Training Time'].append(training_time)
        # wandb.log({'epoch': epoch, 'Loss': running_loss, 'recall@10':recall.item(), 'ndcg@10': ndcg.item()})
        
        if best_score < recall : 
            best_score = recall
            print(f"New best model for Recall@10 : {recall:.4}! saving the best model..")
            torch.save(model.state_dict(), f"{save_dir_path}/best.pth")
    
    else : 
        # save results in dict
        results['Epoch'].append(epoch)
        results['Loss'].append(running_loss)
        results['Valid_Loss'].append(None)
        results['Recall'].append(None)
        # results['NDCG'].append(None)
        results['Training Time'].append(training_time)
        # wandb.log({'epoch': epoch, 'Loss': running_loss})
    print("="*104)
    if should_stop == True: break

# -- save
torch.save(model.state_dict(), f"{save_dir_path}/last.pth")