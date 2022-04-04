import argparse, torch, os
import numpy as np
import pandas as pd

from importlib import import_module

from models import ngcf, lightgcn
from dataset import Data, BaseDataset
from trainers import inference_model

from torch.utils.data import DataLoader

from utils import set_seed, check_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default='42', help=' ')

    parser.add_argument('--model', type=str, default='NGCF', help='model type (default: NGCF)')
    parser.add_argument("--epochs", type=int, default=3000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help=" ")

    parser.add_argument("--emb_dim", type=int, default=64, help=" ")
    parser.add_argument("--layers", nargs="+", type=int, default=[64, 64, 64], help=" ")
    parser.add_argument("--reg", type=int, default=1e-5, help="default: 1e-5")

    parser.add_argument("--node_dropout", type=float, default=0.1, help=" ")
    parser.add_argument("--mess_dropout", type=float, default=0.1, help=" ")

    parser.add_argument("--k", type=int, default=10, help=" ")

    parser.add_argument("--data_dir", type=str, default='../data/', help=" ")
    parser.add_argument("--output_dir", type=str, default='./output', help=" ")

    parser.add_argument("--dataset", type=str, default='', help=" ")

    parser.add_argument("--wandb_name", type=str, default='NGCF', help=" ")
    parser.add_argument("--checkpoint", type=str, default='latest', help=" ")
    parser.set_defaults(save_results=True)

    args = parser.parse_args()

    # -- seed
    set_seed(args.seed)

    # -- checkpoint; check checkpoint and path
    # 특정한 값을 지정해주지 않는다면, 가장 최근 저장되어있는 것을 불러오면 된다.
    # in detail, 현재 추론에 사용할 모델명을 기반으로 저장되어있는 가장 최근의 pt 파일을 가지고 오는 것이다. 
    if args.checkpoint == 'latest':
        latest_save_pt = max([pt_files for pt_files in os.listdir(args.output_dir) if args.model in pt_files])
        checkpoint_path = os.path.join(args.output_dir, latest_save_pt)
        print(f"... Loaded the LATEST Saved pt. PATH: [ {checkpoint_path} ] (model: {args.model}) ...")

    # -- device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    ##### model pre-set #####
    # generate the Normalized-adjacency matrix
    GnnDataset = BaseDataset(path= os.path.join(args.data_dir + args.dataset))
    adj_mtx = GnnDataset.get_adj_mat()

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
        device,
        mode = 'submission'
    ).to(device) # model.cuda()
    # model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Load model from {checkpoint_path} for submission!")

    # -- prediction
    results = inference_model(
        u_emb = model.u_final_embeddings.detach(), 
        i_emb = model.i_final_embeddings.detach(), 
        Rtr = GnnDataset.R_train, 
        Rte = GnnDataset.R_test, 
        k = args.k, 
        device = device
        )


    item_ids = GnnDataset.ratings_df["item"].unique()
    user_ids = GnnDataset.ratings_df["user"].unique()
    idx2item = pd.Series(data=item_ids, index=np.arange(len(item_ids))) 

    result = []
    for idx, items in enumerate(results.cpu().numpy()):
        for item in items:
            result.append((user_ids[idx], idx2item[item]))


    # -- save; submission.csv file 
    submission_path = os.path.join(args.output_dir, "submissions")
    check_path(submission_path)
    file_name = 'submission-' + args.model
    file_ext = '.csv'
    final_path = os.path.join(submission_path, f"{file_name}{file_ext}")

    uniq=1
    while os.path.exists(final_path):
        final_path = os.path.join(submission_path, f"{file_name}-{uniq}{file_ext}")
        uniq += 1

    pd.DataFrame(result, columns=["user", "item"]).to_csv(final_path, index=False)


if __name__ == "__main__":
    main()