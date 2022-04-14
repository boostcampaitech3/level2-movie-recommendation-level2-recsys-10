import argparse
import os
import wandb
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import WDDataset
from models import WideDeep
from trainers import FinetuneTrainer
from tqdm import tqdm
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", default="WideDeep", type=str)
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.1, help="hidden dropout p"
    )
    parser.add_argument("--embedding_dim" , type=int, default=10, help="embedding vector dim")

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=5000, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=2000, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--wandb_name", type=str, help="wandb name")

    # 1. wandb init
    #wandb.init(project="movierec_train_styoo", entity="styoo", name="SASRec_WithPretrain")
    args = parser.parse_args()
    wandb.init(project="movierec_train", entity="egsbj")
    wandb.run.name = args.wandb_name

    # 2. wandb config
    wandb.config.update(args)
    print(str(args))

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "train_ratings.csv"
    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    
    if args.model_name == 'SASRec':
        train_dataset = SASRecDataset(args, user_seq, data_type="train")
        eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
        test_dataset = SASRecDataset(args, user_seq, data_type="test")

    elif args.model_name == 'BERT4Rec':
        train_dataset = BERT4RecDataset(args, user_seq, data_type="train")
        eval_dataset = BERT4RecDataset(args, user_seq, data_type="valid")
        test_dataset = BERT4RecDataset(args, user_seq, data_type="test")


    print("datasets making...")
    train_dataset = WDDataset(args, data_type="train")
    # RandomSampler : Batch를 뽑을때 섞으니, User들의 순서를 섞는것
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size , pin_memory=True
    )
    # train_dataloader = DataLoader(
    #     train_dataset, batch_size= args.batch_size, shuffle=True , pin_memory=True
    # )
    eval_dataset = WDDataset(args,  data_type="valid")
    # SequentialSampler : 순서대로 뽑아야 채점이 편하기에 순서대로 뽑는다, 순서대로 뽑지 않으면 유저 아이디를 매핑해야
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
         eval_dataset, sampler=eval_sampler, batch_size=args.batch_size , pin_memory=True
    )
    # test는 구현 필요
    # test_dataset = WDDataset(args,  data_type="test")
    # test_sampler = SequentialSampler(test_dataset)
    # test_dataloader = DataLoader(
    #     test_dataset, sampler=test_sampler, batch_size=args.batch_size
    # )

    # for input_dims
    df = pd.read_csv(args.data_file)
    n_users = len(df['user'].unique())
    n_items = len(df['item'].unique())
    n_genres = 18
    

    # model = DeepFM(input_dims, embedding_dim, mlp_dims=[30, 20, 10]).to(device)
    
    input_dims = [n_users, n_items, n_genres, n_items, n_genres]
    device = torch.device('cuda')
    model = WideDeep(input_dims=input_dims, mlp_dims=[1024,512,256], args=args).to(device)
    
    # -- trainer -- # 
    trainer = FinetuneTrainer(
        model, train_dataloader, eval_dataloader, None,  None, args
    )
    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)

    for epoch in range(args.epochs):
        trainer.train(epoch)

        scores, precsision, recall  = trainer.valid(epoch)

        wandb.log({"acc" :  scores,
                   "precsision " : precsision,
                   "recall " : recall})

        # early_stopping(np.array(scores[-1:]), trainer.model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break


    # # load the best model
    # trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    # socres, result_info = trainer.test(0)
    # print(result_info)


    # ----  기존 train 방식 ---- #
    # bce_loss = nn.BCELoss()

    # lr , num_epochs = 0.01, 10
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # print("start training..")
    # for e in tqdm(range(num_epochs)):
    #     if num_epochs % 10 == 0:
    #         print("epochs : ", e)
    #     for x, y in train_dataloader:
    #         # x.shape : [256, 39] , y.shape : [256, 1]
    #         x, y = x.to(device), y.to(device)
    #         model.train()
    #         optimizer.zero_grad()
    #         output = model(x)
    #         loss = bce_loss(output, y.float())
    #         loss.backward()
    #         optimizer.step()




    # trainer = FinetuneTrainer(
    #     model, train_dataloader, eval_dataloader, test_dataloader, None, args
    # )

    # # pretrain을 쓸지 안쓸지 확인
    # print(args.using_pretrain)
    # if args.using_pretrain:
    #     pretrained_path = os.path.join(args.output_dir, "Pretrain.pt")
    #     try:
    #         trainer.load(pretrained_path)
    #         print(f"Load Checkpoint From {pretrained_path}!")

    #     except FileNotFoundError:
    #         print(f"{pretrained_path} Not Found! The Model is same as SASRec")
    # else:
    #     print("Not using pretrained model. The Model is same as SASRec")

    # early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
    # for epoch in range(args.epochs):
    #     model.train()

    #     scores, _ = model.valid(epoch)

    #     wandb.log({"recall@5" : scores[0],
    #                "ndcg@5" : scores[1],
    #                "recall@10" : scores[2],
    #                "ndcg@10" : scores[3]})

    #     early_stopping(np.array(scores[-1:]), trainer.model)
    #     if early_stopping.early_stop:
    #         print("Early stopping")
    #         break

    # trainer.args.train_matrix = test_rating_matrix
    # print("---------------Change to test_rating_matrix!-------------------")
    # # load the best model
    # trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    # scores, result_info = trainer.test(0)
    # print(result_info)


if __name__ == "__main__":
    main()
