import argparse
import os
import wandb
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import AutoRecDataset
from models import AutoRec
from trainers import AutoRecTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
    train_valid_split,
    make_inter_mat,
    generate_submission_file,
    negative_sampling,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # data args
    parser.add_argument("--neg_sampling", action="store_true")
    parser.add_argument("--neg_sampling_method", type=str, default = "n_neg") # n_neg, sample_num
    parser.add_argument(
        "--n_negs", type=int, default=1, help="negative sample n_negs"
    )
    parser.add_argument(
        "--neg_sample_num", type=int, default=50, help="negative sample num"
    )

    # model args
    parser.add_argument("--model_name", default="AutoRec", type=str)
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="hidden size of Autorec"
    )
    parser.add_argument(
        "--hidden_activation", type=str, default='sigmoid', help="Autorec encoder activation function"
    )
    parser.add_argument(
        "--out_activation", type=str, default='none', help="Autorec decoder activation function"
    )
    parser.add_argument(
        "--num_layers", type=int, default=1, help="number of layers"
    )
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument(
        "--dropout_rate", type=float, default=0.05, help="dropout rate"
    )

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="early stopping patience"
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
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="sgd momentum"
    )

    # scheduler
    parser.add_argument(
        "--scheduler_factor", type=float, default=0.1, help="scheduler factor"
    )
    parser.add_argument(
        "--scheduler_eps", type=float, default=1e-09, help="scheduler epsilon"
    )
    parser.add_argument(
        "--scheduler_patience", type=int, default=5, help="scheduler_patience"
    )

    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    # parser.add_argument("--candidate_K", type=int, default=30, help="candidate K")
    
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--wandb", action="store_true")

    # 1. wandb init
    #wandb.init(project="movierec_train_styoo", entity="styoo", name="SASRec_WithPretrain")
    args = parser.parse_args()
    if args.wandb:
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
    args_str = f"{args.model_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")

    # save model
    checkpoint = args.wandb_name + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    print('train valid split')
    train_set, valid_set, item_set = train_valid_split(args)

    print('make csr matrix')
    train_mat, valid_mat, item_mat = make_inter_mat(args.data_file, args.model_name, train_set, valid_set, item_set)

    args.train_matrix = train_mat
    
    # print(f'negative sampling : {args.neg_sampling}')
    if args.neg_sampling:
        train_neg_set, item_neg_set = negative_sampling(args, train_set, valid_set, item_set)

        train_neg_mat, item_neg_mat = make_inter_mat(args.data_file, args.model_name, train_neg_set, item_neg_set)

    if args.model_name == 'AutoRec':
        if args.neg_sampling:
            train_dataset = AutoRecDataset(args, item_neg_mat, valid_mat)
            eval_dataset = AutoRecDataset(args, train_neg_mat, valid_mat)
            # test_dataset = AutoRecDataset(args, item_mat, None)
            submission_dataset = AutoRecDataset(args, item_neg_mat, valid_mat)
        else:
            train_dataset = AutoRecDataset(args, item_mat, valid_mat)
            eval_dataset = AutoRecDataset(args, train_mat, valid_mat)
            # test_dataset = AutoRecDataset(args, item_mat, None)
            submission_dataset = AutoRecDataset(args, item_mat, valid_mat)


    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle = True, pin_memory = True
    )

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle = False, pin_memory = True
    )

    # test_sampler = RandomSampler(test_dataset)
    # test_dataloader = DataLoader(
    #     test_dataset, sampler=test_sampler, batch_size=args.batch_size, shuffle = False
    # )

    submission_dataloader = DataLoader(
        submission_dataset, batch_size=args.batch_size, shuffle = False, pin_memory = True
    )

    if args.model_name == 'AutoRec':
        args.input_dim = args.train_matrix.shape[1]

        model = AutoRec(args=args)

        trainer = AutoRecTrainer(
            model, train_dataloader, eval_dataloader, None, submission_dataloader, args
        )
    
    print(model)
    
    
    early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)

    for epoch in tqdm(range(args.epochs)):
        trainer.train(epoch)

        scores, _ = trainer.valid(epoch)
        
        # 3. wandb log
        if args.wandb:
            wandb.log({"recall@5" : scores[0],
                    "ndcg@5" : scores[1],
                    "recall@10" : scores[2],
                    "ndcg@10" : scores[3]})

        early_stopping(np.array([scores[2]]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # trainer.args.train_matrix = test_rating_matrix
    # print("---------------Change to test_rating_matrix!-------------------")
    # load the best model
    trainer.args.train_matrix = item_mat
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    preds = trainer.submission(0)

    generate_submission_file(args.data_file, preds, args.model_name)
    # scores, result_info = trainer.test(0)
    # print(result_info)


if __name__ == "__main__":
    main()
