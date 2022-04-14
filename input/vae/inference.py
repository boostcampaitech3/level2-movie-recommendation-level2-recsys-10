import argparse, torch, os
from cmath import exp
import numpy as np
import pandas as pd
import bottleneck as bn
from importlib import import_module
from torch.utils.data import DataLoader
from utils import set_seed, check_path
from dataset import BaseDataset, ValidDataset
    
    
parser = argparse.ArgumentParser()

# env parameter
parser.add_argument('--seed', type=int, default='42', help=' ')
parser.add_argument('--dataset', type=str)
parser.add_argument('--k', type=int, default=10)

# model parameter
parser.add_argument('--model', type=str, default='RecVAE', help='model type (default: RecVAE)')
parser.add_argument('--hidden_dim', type=int, default=600)
parser.add_argument('--latent_dim', type=int, default=200)
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--gamma', type=float, default=0.0035)
parser.add_argument('--dropout_rate', type=float, default=0.5)

# path
parser.add_argument("--data_dir", type=str, default='../data/', help=" ")
parser.add_argument("--output_dir", type=str, default='./output', help=" ")
parser.add_argument("--checkpoint", type=str, default='RecVAE50-gamma0035', help=" ")

args = parser.parse_args()

# -- seed
set_seed(args.seed)

# -- checkpoint; check checkpoint and path
# 특정한 값을 지정해주지 않는다면, 가장 최근 저장되어있는 것을 불러오면 된다.
# in detail, 현재 추론에 사용할 모델명을 기반으로 저장되어있는 가장 최근의 pt 파일을 가지고 오는 것이다. 
checkpoint_path = os.path.join(args.output_dir, args.checkpoint)
# checkpoint_path = os.path.join(checkpoint_path, 'last.pth') #best.pth
if os.path.exists(os.path.join(checkpoint_path, 'best.pth')) : 
    checkpoint_path = os.path.join(checkpoint_path, 'best.pth')
else : 
    checkpoint_path = os.path.join(checkpoint_path, 'last.pth')


print(f"... Loaded the Saved pt. PATH: [ {checkpoint_path} ] (model: {args.model}) ...")
print()

# -- device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# -- submission path

# -- dataset
submission_dataset = BaseDataset(path = args.data_dir, mode='train_all')
submission_loader = DataLoader(submission_dataset, batch_size=2000, drop_last=False, shuffle=False, pin_memory=use_cuda)

# -- model state load
model_module = getattr(import_module("models"), args.model)  # default: RecVAE
model = model_module(
    hidden_dim = args.hidden_dim, 
    latent_dim = args.latent_dim, 
    input_dim = submission_dataset.n_items,
    dropout_rate = args.dropout_rate
).to(device)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print(f"Load model from {checkpoint_path} for submission!")


# -- get result
with torch.no_grad():
    model.eval()
    for batch_index, batch_data in enumerate(submission_loader):
        X_train = batch_data.to(device)
        X_pred, mu, logvar = model(X_train)

        X_pred = X_pred.cpu().numpy()
        X_train = X_train.cpu().numpy()
        # 상위 k 개를 밀어 넣는다

        X_pred[X_train>0] = 0
        ind = np.argpartition(X_pred, -args.k)[:, -args.k:] # 우리가 하는 Task에서는 ind = item number (물론 이후에 또 바꿔줘야한다.)

        if batch_index == 0 : 
            pred_list = ind
        else :  
            pred_list = np.concatenate([pred_list, ind])


idx2user = {v:k for k,v in dict(submission_dataset.user2idx).items()}
idx2item = {v:k for k,v in dict(submission_dataset.item2idx).items()}

pred = pd.concat({k: pd.Series(v) for k, v in enumerate(pred_list)}).reset_index(0)
pred.columns = ['user', 'item']
pred['user'] = pred['user'].replace(idx2user)
pred['item'] = pred['item'].replace(idx2item)


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

pred.to_csv(final_path, index=False)
print(f"Saved as {final_path}")


