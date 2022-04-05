import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy import sparse

from datasets import MultiVAEDataset
from models import MultiVAE
from utils import NDCG_binary_at_k_batch, Recall_at_k_batch, loss_function_vae
from preprocessing import MultiVAE_preprocess, numerize

import wandb

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

def evaluate(args, model, criterion, data_tr, data_te):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    n100_list = []
    r20_list = []
    r50_list = []
    r10_list = []
    
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(device)
              
            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 
                            1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            recon_batch, mu, logvar = model(data_tensor)

            loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)

            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)
            r10 = Recall_at_k_batch(recon_batch, heldout_data, 10)

            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)
            r10_list.append(r10)
 
    total_loss /= len(range(0, e_N, args.batch_size))
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)
    r10_list = np.concatenate(r10_list)

    return total_loss, np.mean(n100_list), np.mean(r20_list), np.mean(r50_list), np.mean(r10_list)

def submission_multi_vae(args, model):
    print("-----submission-----")
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

    test_data_tensor = naive_sparse2tensor(data).to(device)

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
        MultiVAE_preprocess(args)

        loader = MultiVAEDataset(args.data)

        n_items = loader.load_n_items()
        train_data = loader.load_data('train')
        vad_data_tr, vad_data_te = loader.load_data('validation')
        test_data_tr, test_data_te = loader.load_data('test')

        global N
        N = train_data.shape[0]
        idxlist = list(range(N))

        p_dims = [200, 600, n_items]
        model = MultiVAE(args=args, p_dims=p_dims).to(device)
        wandb.config.update(args)

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)

        criterion = loss_function_vae

        # train
        best_r10 = -np.inf
        global update_count
        update_count = 0

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            model.train()
            train_loss = 0.0
            start_time = time.time()

            N = train_data.shape[0]
            idxlist = list(range(N))

            np.random.shuffle(idxlist)
            
            for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
                end_idx = min(start_idx + args.batch_size, N)
                data = train_data[idxlist[start_idx:end_idx]]
                data = naive_sparse2tensor(data).to(device)
                optimizer.zero_grad()

                if args.total_anneal_steps > 0:
                    anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
                else:
                    anneal = args.anneal_cap

                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                
                loss = criterion(recon_batch, data, mu, logvar, anneal)

                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                update_count += 1

                if batch_idx % args.log_interval == 0 and batch_idx > 0:
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                            'loss {:4.2f}'.format(
                                epoch, batch_idx, len(range(0, N, args.batch_size)),
                                elapsed * 1000 / args.log_interval,
                                train_loss / args.log_interval))

                    start_time = time.time()
                    train_loss = 0.0

            val_loss, n100, r20, r50, r10 = evaluate(args, model, criterion, vad_data_tr, vad_data_te)

            wandb.log({"valid loss" : val_loss,
                "n100" : n100,
                "r20" : r20, 
                "r10" : r10,
                "r50" : r50})
            n_iter = epoch * len(range(0, N, args.batch_size))

            # Save the model if the n100 is the best we've seen so far.
            if r10 > best_r10:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_r10 = r10

        # Load the best saved model.
        with open(args.save, 'rb') as f:
            model = torch.load(f)

        # Run on test data.
        test_loss, n100, r20, r50, r10 = evaluate(args, model, criterion, test_data_tr, test_data_te)
        print('=' * 89)
        print('| End of training | test loss {:4.2f} | n100 {:4.2f} | r20 {:4.2f} | r10 {:4.2f} |'
                'r50 {:4.2f}'.format(test_loss, n100, r20, r10 ,r50))
        print('=' * 89)

    submission_multi_vae(args, model)

if __name__ == "__main__":
    main()