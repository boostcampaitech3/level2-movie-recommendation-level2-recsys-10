import argparse, os, torch
from ast import parse
from time import time

import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,CosineAnnealingLR
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

from utils import precision, recall, set_seed, f1_score
from models.deepfm import DeepFM
from dataset import DeepFMDataset

import wandb
from tqdm import tqdm


def main():
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default='42', help=' ')
    
    parser.add_argument('--v', type=str, default='1')
    parser.add_argument('--data_dir', type=str, default='../data/train')
    parser.add_argument("--output_dir", default="./output/", type=str)
    parser.add_argument('--model', type=str, default='deepfm', help='Model Name (deepfm)')
    parser.add_argument('--valid_size', type=int, default=10, help='valid size per user')
    parser.add_argument('--mlp_dims', nargs='+', type=int, default=[200, 200, 200], help = 'Multi-Layer-Perceptron dimensions list')
    parser.add_argument('--embedding_dim', type=int, default=400, help='embedding_dim for input tensor')
    parser.add_argument('--drop_rate', type=float, default=0.6, help='Drop rate')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument("--lr_decay_step", type=int, default=100, help="default: 200") 
    parser.add_argument("--gamma", type=float, default=0.1, help="default: 0.1") 
    parser.add_argument('--epoch', type=int, default=5000, help='num of epochs')
    
    parser.add_argument("--wandb_name", type=str, default='-', help=" ")
    parser.add_argument("--save_results", action="store_true")
    args = parser.parse_args()

    ##### wandb init #####
    wandb.init(project="movierec_train", entity="egsbj")
    wandb.run.name = args.model + args.wandb_name
    wandb.config.update(args)

    # seed & device
    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.model == 'deepfm':
        print("Dataset setting") 
        ############
        rating = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'))
        genres = pd.read_csv(os.path.join(args.data_dir, 'genres.tsv'), sep='\t')
        year = pd.read_csv(os.path.join(args.data_dir,'years.tsv'), sep='\t')
        # year 결측치 처리
        not_year = list(set(rating['item']) - set(year['item']))
        not_year.sort()
        null_year = [1921,1920,1919,1915,1916,1917,1902,2015]
        missing_year = pd.DataFrame({'item':not_year, 'year':null_year})
        year = pd.concat([year, missing_year])
        
        users = list(set(rating.loc[:,'user']))
        users.sort()
        items =  list(set((rating.loc[:, 'item'])))
        items.sort()
        if len(users)-1 != max(users):
            users_dict = {users[i]: i for i in range(len(users))}
            rating['user']  = rating['user'].map(lambda x : users_dict[x])

        if len(items)-1 != max(items):
            items_dict = {items[i]: i for i in range(len(items))}
            rating['item']  = rating['item'].map(lambda x : items_dict[x])
            year['item']  = year['item'].map(lambda x : items_dict[x])
            genres['item']  = genres['item'].map(lambda x : items_dict[x])

        le = LabelEncoder()
        year['year'] = le.fit_transform(year['year'])
        
        year = {i:v for i,v in zip(year['item'], le.fit_transform(year['year']))}
        
        # genre Multi Label Binarize 
        genre_dict = {genre:i for i, genre in enumerate(set(genres['genre']))}
        genres['genre'] = genres['genre'].map(lambda x : genre_dict[x])
        genres = genres.groupby('item')['genre'].agg(lambda x : list(x))
        mlb = MultiLabelBinarizer()
        multi_genres = mlb.fit_transform(genres.values)
        genres = {i:v for i,v in zip(genres.index, multi_genres)}

        #####################################
        valid_size = args.valid_size
        train_dataset = DeepFMDataset(rating, year, genres, valid_size, mode = 'seq')
        print("train setting done")
        test_dataset = DeepFMDataset(rating, year, genres, valid_size, mode = 'static')
        print("valid setting done")
        n_user,n_item,n_year = train_dataset.get_num_context()

        train_loader = DataLoader(train_dataset, batch_size=3920, shuffle=True, num_workers=3)
        test_loader = DataLoader(test_dataset, batch_size=3920, shuffle=True, num_workers=3)
        print("Dataset setting done")

        #####################################
        # config setting
        input_dims = [n_user, n_item, n_year]
        embedding_dim = args.embedding_dim
        mlp_dims = args.mlp_dims
        model = DeepFM(input_dims, embedding_dim, mlp_dims=mlp_dims).to(device)
        bce_loss = nn.BCELoss() # Binary Cross Entropy loss

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=args.gamma)
        # scheduler = ReduceLROnPlateau(optimizer, patience=100,mode='min')
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
        losses = [100]
        
        print("Training Strat")
        for e in tqdm(range(args.epoch)) :
            t1 = time()
            model.train()
            
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                output = model(x)
                loss = bce_loss(output, y.float())
                loss.backward()
                optimizer.step()

            scheduler.step(loss)
            losses.append(loss)
            training_time = time()-t1
            print(f"[EPOCH: {e :3}/{args.epoch}] [Train] time: {training_time:4.2}s | Loss: {loss:4.4}", end=' | ')
            print("[Learning Rate]: {}".format(scheduler.optimizer.state_dict()['param_groups'][0]['lr']))
            # 모델 저장
            if loss == min(losses):
                save_path = os.path.join(args.output_dir,f'DeepFM_{args.v}.pt')
                torch.save(model.state_dict(), save_path)
                print(f'Model Saved, DeepFM_{args.v}.pt')

            # 10 epoch 마다 val
            if e % 10 == 0:
                correct_result_sum = 0
                f1, precision, recall= [],[],[]
                
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    model.eval()
                    output = model(x)
                    result = torch.round(output)
                    correct_result_sum += (result == y).sum().float()
                    f1_, precision_, recall_ = f1_score(result, y)
                    f1.append(f1_)
                    precision.append(precision_)
                    recall.append(recall_)
                acc = correct_result_sum/len(test_dataset)*100
                f1 = sum(f1)/len(f1)
                precision = sum(precision)/len(precision)
                recall = sum(recall)/len(recall)
                wandb.log({'epoch': e, 'Loss': loss, 'Acc': acc, 'F1':f1, 'Recall':recall, 'Precision':precision})
                print("Acc : {:.2f}%".format(acc.item()), end = '\t')
                print("F1-score : {:.2f}%".format(f1*100), end = '\t')
                print("Recall : {:.2f}%".format(recall*100), end = '\t')
                print("Precision : {:.2f}%".format(precision*100))

                
if __name__ == '__main__':
    main()

