import argparse
import os
import wandb
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from models import MF
from utils import (
    check_path,
    get_user_seqs,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", default="MF", type=str)
    parser.add_argument(
        "--hidden_size", type=int, default=100, help="hidden size of latent vector"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.5,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p"
    )
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--mask_p", type=float, default=0.15, help="mask probability")
    parser.add_argument("--rm_position", action="store_true", help="remove position embedding")
    parser.add_argument("--l2_reg", default=0.2, type=float, help='l2 regularization')

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
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

    parser.add_argument("--using_pretrain", action="store_true")
    parser.add_argument("--patience", default = 10, type=int, help="early stopping patience")

    parser.add_argument("--wandb_name", type=str)

    # 1. wandb init
    # wandb.init(project="movierec_train_styoo", entity="styoo", name="SASRec_WithPretrain")
    args = parser.parse_args()
    wandb.init(project="movierec_train", entity="egsbj")
    wandb.run.name = args.wandb_name

    # # 2. wandb config
    wandb.config.update(args)
    print(str(args))

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "train_ratings.csv"

    user_seq, max_item, _, _, sparse_matrix = get_user_seqs(
        args.data_file, args.model_name
    )


    args.item_size = max_item

    # save model args
    args_str = f"{args.model_name}" #-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")

    args.train_matrix = sparse_matrix
    
    if args.model_name == 'MF':
        model = MF(args)

    rmse_min = float("inf")
    early_stopping_count = 0

    for epoch in tqdm(range(args.epochs)):
        rmse = model.train()
        
        print("epoch: %d, error = %.4f" % (epoch, rmse))
        # 3. wandb log
        wandb.log({"rmse" : rmse})

        if rmse < rmse_min:
            print(f"rmse min : {rmse}")
            rmse_min = rmse
            early_stopping_count = 0
        else:
            early_stopping_count += 1
            print(f"EarlyStopping counter: {early_stopping_count} out of {args.patience}")
            if early_stopping_count >= args.patience:
                break

    pred_list = model.submission()

    generate_submission_file(args.data_file, pred_list, args.model_name)


if __name__ == "__main__":
    main()
