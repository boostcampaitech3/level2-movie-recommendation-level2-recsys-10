from importlib.resources import path
import torch, random
import numpy as np
from time import time
from datetime import datetime
from pytz import timezone

from copy import deepcopy
from utils import set_seed, get_lr, early_stopping, Recall_at_k_batch, NDCG_binary_at_k_batch
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
parser.add_argument('--hidden_dim', type=int, default=600)
parser.add_argument('--latent_dim', type=int, default=200)
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--gamma', type=float, default=0.005)

# train parameter (with valid)
parser.add_argument("--epochs", type=int, default=3000, help="number of epochs")

parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--n-enc_epochs', type=int, default=3)
parser.add_argument('--n-dec_epochs', type=int, default=1)
parser.add_argument('--not-alternating', type=bool, default=False)
parser.add_argument("--eval_N", type=int, default=1, help=" ")
parser.add_argument("--k", type=int, default=10, help=" ")

# path
parser.add_argument("--data_dir", type=str, default='../data/', help=" ")
parser.add_argument("--output_dir", type=str, default='./output', help=" ")

# utilities
parser.add_argument("--wandb_name", type=str, default='-', help=" ")
parser.add_argument("--checkpoint", type=str, default='latest', help=" ")

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
results = {"Epoch": [], "Loss": [], "Valid_Loss":[], "Recall": [], "NDCG": [], "Training Time": []}

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

    return loss

###### train & validation ##### 
running_loss = 0.0
for epoch in range(args.epochs):
    # -- train
    t1 = time()

    if args.not_alternating:
        model.dropout_rate = 0.5
        loss = trainer(model, opts=[optimizer_encoder, optimizer_decoder], n_epochs=1)
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
    print(f"[EPOCH: {epoch:3}/{args.epochs}] [Train] time: {training_time:4.2}s | Loss: {running_loss:4.4} | lr: {current_lr}")

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

                _recall = Recall_at_k_batch(recon_batch, label_data, args.k)
                _ndcg = NDCG_binary_at_k_batch(recon_batch, label_data, args.k) # TODO 현재 데이터 셋에 맞게 수정해준다. 

                recall_list.append(_recall)
                ndcg_list.append(_ndcg)

        total_loss /= (train_dataset.n_users//args.batch_size) 
        recall = np.concatenate(recall_list).mean()
        ndcg = np.concatenate(ndcg_list).mean()
        print(f"---------------------------------------------------------------------------------------------------------------------")
        print(f"[Valid] time: {time()-t2:4.2}s, ", end='')
        print(f"Valid Loss: {total_loss:4.4} | Recall@{args.k}: {recall:.4} | NDCG@{args.k}: {ndcg:.4}")
        print(f"----------------------------------------------------------------------------------------------------------------------")
        cur_best_metric, stopping_step, should_stop = early_stopping(recall, cur_best_metric, stopping_step, flag_step=10)

        # save results in dict
        results['Epoch'].append(epoch)
        results['Loss'].append(running_loss)
        results['Valid_Loss'].append(total_loss)
        results['Recall'].append(recall.item())
        results['NDCG'].append(ndcg.item())
        results['Training Time'].append(training_time)
        # wandb.log({'epoch': epoch, 'Loss': running_loss, 'recall@10':recall.item(), 'ndcg@10': ndcg.item()})
        
        # ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
        # TODO BEST SCORE 기준으로 저장, 마지막 iteration으로 자동 저장하는 기능 추가 ##
        # ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
    
    else : 
        # save results in dict
        results['Epoch'].append(epoch)
        results['Loss'].append(running_loss)
        results['Valid_Loss'].append(None)
        results['Recall'].append(None)
        results['NDCG'].append(None)
        results['Training Time'].append(training_time)
        # wandb.log({'epoch': epoch, 'Loss': running_loss})

    if should_stop == True: break


# TODO -1 : alternating 학습 방법으로 전체 학습 구성하기 -> trainers.py 로 새로운 훈련함수를 정의해주자.

# TODO -2 : (참고로, 여기에서는 testset은 따로 사용하지 않기때문에 그것은 가만히 놔둔다.)


# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
# def run(model, opts, train_data, batch_size, n_epochs, beta, gamma, dropout_rate):
#     model.train()
#     for epoch in range(n_epochs):
#         for batch in generate(batch_size=batch_size, device=device, data_in=train_data, shuffle=True):
#             ratings = batch.get_ratings_to_dev()

#             for optimizer in opts:
#                 optimizer.zero_grad()
                
#             _, loss = model(ratings, beta=beta, gamma=gamma, dropout_rate=dropout_rate)
#             loss.backward()
            
#             for optimizer in opts:
#                 optimizer.step()

# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
# for epoch in range(args.n_epochs):

#     if args.not_alternating:
#         run(opts=[optimizer_encoder, optimizer_decoder], n_epochs=1, dropout_rate=0.5, **learning_kwargs)
#     else:
#         run(opts=[optimizer_encoder], n_epochs=args.n_enc_epochs, dropout_rate=0.5, **learning_kwargs)
#         model.update_prior()
#         run(opts=[optimizer_decoder], n_epochs=args.n_dec_epochs, dropout_rate=0, **learning_kwargs)

#     train_scores.append(
#         evaluate(model, train_data, train_data, metrics, 0.01)[0]
#     )
#     valid_scores.append(
#         evaluate(model, valid_in_data, valid_out_data, metrics, 1)[0]
#     )
    
#     if valid_scores[-1] > best_ndcg:
#         best_ndcg = valid_scores[-1]
#         model_best.load_state_dict(deepcopy(model.state_dict()))
        

#     print(f'epoch {epoch} | valid ndcg@100: {valid_scores[-1]:.4f} | ' +
#           f'best valid: {best_ndcg:.4f} | train ndcg@100: {train_scores[-1]:.4f}')

# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
# model_kwargs = {
#     'hidden_dim': args.hidden_dim,
#     'latent_dim': args.latent_dim,
#     'input_dim': train_data.shape[1]
# }
# metrics = [{'metric': ndcg, 'k': 100}]

# best_ndcg = -np.inf
# train_scores, valid_scores = [], []

# model = VAE(**model_kwargs).to(device)
# model_best = VAE(**model_kwargs).to(device)

# learning_kwargs = {
#     'model': model,
#     'train_data': train_data,
#     'batch_size': args.batch_size,
#     'beta': args.beta,
#     'gamma': args.gamma
# }

# decoder_params = set(model.decoder.parameters())
# encoder_params = set(model.encoder.parameters())

# optimizer_encoder = optim.Adam(encoder_params, lr=args.lr)
# optimizer_decoder = optim.Adam(decoder_params, lr=args.lr)

    
# test_metrics = [{'metric': ndcg, 'k': 100}, {'metric': recall, 'k': 20}, {'metric': recall, 'k': 50}]

# final_scores = evaluate(model_best, test_in_data, test_out_data, test_metrics)

# for metric, score in zip(test_metrics, final_scores):
#     print(f"{metric['metric'].__name__}@{metric['k']}:\t{score:.4f}")