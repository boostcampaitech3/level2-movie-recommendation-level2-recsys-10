import argparse, os, torch
from ast import parse
from time import time

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from utils import set_seed
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
    parser.add_argument("--output_dir", default="../../output/", type=str)
    parser.add_argument('--model', type=str, default='deepfm', help='Model Name (deepfm)')
    parser.add_argument('--mlp_dims', nargs='+', type=int, default=[30,20,10], help = 'Multi-Layer-Perceptron dimensions list')
    parser.add_argument('--embedding_dim', type=int, default=10, help='embedding_dim for input tensor')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='Drop rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument("--lr_decay_step", type=int, default=50, help="default: 500") 
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
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.model == 'deepfm':
        print("Dataset setting")
        dataset = DeepFMDataset(os.path.join(args.data_dir, 'deepfm_train.json'))
        n_user,n_item,n_year = dataset.get_num_context()
        train_ratio = 0.9

        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
        print("Dataset setting done")

        #####################################
        # config setting
        input_dims = [n_user, n_item, n_year]
        embedding_dim = args.embedding_dim
        model = DeepFM(input_dims, embedding_dim, mlp_dims=[30, 20, 10]).to(device)
        bce_loss = nn.BCELoss() # Binary Cross Entropy loss

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        losses = [1]
        
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

            scheduler.step()
            losses.append(loss)
            training_time = time()-t1
            print(f"[EPOCH: {e :3}/{args.epoch}] [Train] time: {training_time:4.2}s | Loss: {loss:4.4}")
            
            # 모델 저장
            if loss < min(losses):
                save_path = os.path.join(args.output_dir,f'DeepFM_{args.v}_loss{loss:4.4}.pt')
                torch.save(model.state_dict(), save_path)
                print(f'Model Saved, DeepFM_{args.v}_loss{loss:4.4}.pt')

            # 10 epoch 마다 val
            if e % 10 == 0:
                correct_result_sum = 0
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    model.eval()
                    output = model(x)
                    result = torch.round(output)
                    correct_result_sum += (result == y).sum().float()

                acc = correct_result_sum/len(test_dataset)*100
                wandb.log({'epoch': e, 'Loss': loss, 'Acc': acc})
                print("Acc : {:.2f}%".format(acc.item()))


if __name__ == '__main__':
    main()

