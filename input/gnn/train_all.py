# Use all data for training but not validation section.
# So, Can't find any metrics at this page but You can check training loss

######################
### STILL DEVELOPE ###
######################

import argparse, os, torch
import pandas as pd
from importlib import import_module
from time import time
from datetime import datetime
from pytz import timezone

from tqdm import tqdm

import wandb
from trainers import train, eval_model
from dataset import BaseDataset
from models import ngcf, lightgcn
from utils import early_stopping, set_seed, check_path

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


def main():
    ##### argparser #####
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default='42', help=' ')

    parser.add_argument('--model', type=str, default='NGCF', help='model type (default: NGCF)')
    parser.add_argument("--epochs", type=int, default=3000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help=" ")

    parser.add_argument("--emb_dim", type=int, default=64, help=" ")
    parser.add_argument("--layers", nargs="+", type=int, default=[64, 64, 64], help=" ")
    parser.add_argument("--reg", type=int, default=1e-5, help="default: 1e-5")
    parser.add_argument("--lr", type=float, default=1e-3, help="default: 1e-4") 
    parser.add_argument("--lr_decay_step", type=int, default=500, help="default: 500") 

    parser.add_argument("--node_dropout", type=float, default=0.1, help=" ")
    parser.add_argument("--mess_dropout", type=float, default=0.1, help=" ")

    parser.add_argument("--data_dir", type=str, default='../data/', help=" ")
    parser.add_argument("--output_dir", type=str, default='./output', help=" ")
    parser.add_argument("--dataset", type=str, default='', help=" ")

    parser.add_argument("--wandb_name", type=str, default='NGCF(TrainAll)', help=" ")
    parser.add_argument("--save_results", action="store_true")
    parser.set_defaults(save_results=True)

    args = parser.parse_args()


    ##### wandb init #####
    wandb.init(project="movierec_train", entity="egsbj")
    wandb.run.name = args.wandb_name
    wandb.config.update(args)

    # -- seed 
    set_seed(args.seed)

    # -- device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    GnnDataset = BaseDataset(path= os.path.join(args.data_dir + args.dataset), mode='train_all')
    adj_mtx = GnnDataset.get_adj_mat()

    # -- DataLoader
    train_loader = DataLoader(GnnDataset, batch_size= args.batch_size, pin_memory= use_cuda, shuffle= True)

    # -- model
    model_module = getattr(import_module("models"), args.model)  # default: NGCF
    model = model_module(
        GnnDataset.n_users,
        GnnDataset.n_items,
        args.emb_dim,
        args.layers,
        args.reg,
        args.node_dropout,
        args.mess_dropout,
        adj_mtx,
        device
    ).to(device)
    # model = torch.nn.DataParallel(model)

    # -- optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        # weight_decay=1e-4
    )

    # opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    # optimizer = opt_module(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=args.lr,
    #     weight_decay=5e-4
    # )

    # -- loss & metric
    # criterion = create_criterion(args.criterion)  # default: cross_entropy
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)


    # -- system log
    today = datetime.now(timezone('Asia/Seoul'))
    print("Start at " + str(today))
    print("Using " + str(device) + " for computations")
    print("Params on CUDA: " + str(next(model.parameters()).is_cuda))
    results = {"Epoch": [], "Loss": [], "Training Time": []}

    ################################ train ################################  
    for epoch in range(args.epochs):
        t1 = time()
        model.train()
        running_loss=0
        for idx, train_batch in enumerate(train_loader):
            users, pos_items, neg_items = train_batch
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()
            loss = model(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        training_time = time()-t1
        print(f"[EPOCH: {epoch:3}/{args.epochs}] time: {training_time:4.2}s, Loss: {running_loss:4.4}")
        scheduler.step()

        # save results in dict
        results['Epoch'].append(epoch)
        results['Loss'].append(running_loss)
        results['Training Time'].append(training_time)
        wandb.log({'epoch': epoch, 'Loss': running_loss})

    ################################ save ################################
    if args.save_results:
        date = today.strftime("%Y%m%d_%H%M")

        # save model as .pt file
        check_path(args.output_dir)
        save_file_name = str(date) + "_" + args.model + "_" + args.dataset + "train_all.pt"
        output_dirs = os.path.join(args.output_dir, save_file_name)
        print("Saving the model ... ")
        torch.save(model.state_dict(), output_dirs)

if __name__ == "__main__":
    main()