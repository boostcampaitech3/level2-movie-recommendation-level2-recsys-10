import argparse, os, torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from ast import parse

from utils import set_seed
from models.deepfm import DeepFM
from dataset import DeepFMDataset

import wandb
from tqdm import tqdm


def main():
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default='42', help=' ')
    
    parser.add_argument('--data_dir', type=str, default='../data/train')
    parser.add_argument('--model', type=str, default='deepfm', help='Model Name (deepfm, )')
    parser.add_argument('--mlp_dims', nargs='+', type=int, default=[30,20,10], help = 'Multi-Layer-Perceptron dimensions list')
    parser.add_argument('--embedding_dim', type=int, default=10, help='embedding_dim for input tensor')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='Drop rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epoch', type=int, default=200, help='num of epochs')
    
    parser.add_argument("--wandb_name", type=str, default='-', help=" ")
    parser.add_argument("--save_results", action="store_true")
    args = parser.parse_args()

    ##### wandb init #####
    wandb.init(project="movierec_train", entity="egsbj")
    wandb.run.name = args.model + args.wandb_name
    wandb.config.update(args)

    # seed & device
    set_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    


    if args.model == 'deepfm':

        dataset = DeepFMDataset(os.path.join(args.data_dir, 'deepfm_train.json'))
        n_user,n_item,n_year = dataset.get_num_context()
        train_ratio = 0.7

        train_size = int(train_ratio * len(data))
        test_size = len(data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)

        #####################################
        
        input_dims = [n_user, n_item, n_year]
        embedding_dim = args.embedding_dim
        model = DeepFM(input_dims, embedding_dim, mlp_dims=[30, 20, 10]).to(device)
        bce_loss = nn.BCELoss() # Binary Cross Entropy loss
        lr, num_epochs = args.lr, args.epoch
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for e in tqdm(range(num_epochs)) :
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                model.train()
                optimizer.zero_grad()
                output = model(x)
                loss = bce_loss(output, y.float())
                loss.backward()
                optimizer.step()


if __name__ == '__main__':
    main()

