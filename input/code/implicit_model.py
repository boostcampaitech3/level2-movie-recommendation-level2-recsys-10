import argparse
import os
import wandb
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import implicit

import warnings
warnings.filterwarnings(action='ignore')


from models import Implicit_model
from utils import (
    check_path,
    generate_submission_file,
    get_user_seqs,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", default="ALS", type=str) # ALS, BPR
    parser.add_argument("--hidden_size", type=int, default=100, help="The number of latent factors to compute")
    parser.add_argument("--regularization", type=float, default=0.01, help=" The regularization factor to use")

    # train args
    parser.add_argument("--iterations", type=int, default=15, help="number of epochs")
    parser.add_argument("--calculate_loss", action="store_false", help="calculate train loss")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # parser.add_argument("--wandb_name", type=str)

    # 1. wandb init
    # wandb.init(project="movierec_train_styoo", entity="styoo", name="SASRec_WithPretrain")
    args = parser.parse_args()
    # wandb.init(project="movierec_train", entity="egsbj")
    # wandb.run.name = args.wandb_name

    # # 2. wandb config
    # wandb.config.update(args)
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
    
    print(args.model_name)

    model = Implicit_model(args)

    print(implicit.gpu.HAS_CUDA)
    
    print(f'{args.model_name} train')
    
    model.train()
    
    print(f'{args.model_name} submission')
    
    pred_list = model.submission()
    
    print('submission file generate')
    generate_submission_file(args.data_file, pred_list, args.model_name)


if __name__ == "__main__":
    main()
