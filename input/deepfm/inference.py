import os
import argparse
from xmlrpc.client import boolean

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

from models.deepfm import DeepFM, DeepFM_renew
from dataset import DeepFMDataset_renew, DeepFMDataset

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default='42', help=' ')
    parser.add_argument('--rerank', type=boolean, default=False, help='IF True: re-rank top 50 per user to top10')
    parser.add_argument('--use_all', type=str2bool, default=True, help='num of epochs')

    parser.add_argument('--data_dir', type=str, default='../data/train')
    parser.add_argument("--output_dir", default="./output/", type=str)
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--candidate", type=str, default='submission-RecVAE-top50.csv')

    parser.add_argument('--train_all', type=str2bool, default=True, help='num of epochs')
    parser.add_argument('--mlp_dims', nargs='+', type=int, default=[80, 80, 80], help = 'Multi-Layer-Perceptron dimensions list')
    parser.add_argument('--embedding_dim', type=int, default=200, help='embedding_dim for input tensor')

    args = parser.parse_args()

    ############################################
    ##### 데이터 셋 구성 전처리 #####
    print("Dataset setting") 

    data_dir = "../data/train"

    rating = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'))
    genres = pd.read_csv(os.path.join(args.data_dir, 'genres.tsv'), sep='\t')
    year = pd.read_csv(os.path.join(args.data_dir,'years.tsv'), sep='\t')

    # year 결측치 처리
    not_year = list(set(rating['item']) - set(year['item']))
    not_year.sort()
    null_year = [1921,1920,1919,1915,1916,1917,1902,2015]
    missing_year = pd.DataFrame({'item':not_year, 'year':null_year})
    year = pd.concat([year, missing_year])
    
    director = pd.read_csv(os.path.join(args.data_dir, 'directors.tsv'),sep='\t')
    writer = pd.read_csv(os.path.join(args.data_dir, 'writers.tsv'), sep='\t')
    writer_director = pd.concat([writer,director.rename(columns={'director':'writer'})],axis=0).rename(columns={'writer':'contributor'})
    
    # 유저, 아이템 Encoding
    users = list(set(rating.loc[:,'user']))
    users.sort()
    items =  list(set((rating.loc[:, 'item'])))
    items.sort()
    rating_origin = rating.copy()
    # rerank시 rating_df / submission 이름 변경 
    submission_name = args.model + '_all.csv'
    if args.rerank:
        candidate_file = args.candidate
        rating = pd.read_csv(os.path.join(data_dir, candidate_file))
        submission_name = args.model + 'submission.csv'
        print(f'##Rerank-Mode by {candidate_file} to {submission_name}##')

    if len(users)-1 != max(users):
        users_dict = {users[i]: i for i in range(len(users))}
        rating['user']  = rating['user'].map(lambda x : users_dict[x])
        rating_origin['user']  = rating_origin['user'].map(lambda x : users_dict[x])

    if len(items)-1 != max(items):
        items_dict = {items[i]: i for i in range(len(items))}
        rating['item']  = rating['item'].map(lambda x : items_dict[x])
        rating_origin['item']  = rating_origin['item'].map(lambda x : items_dict[x])
        
        year['item']  = year['item'].map(lambda x : items_dict[x])
        genres['item']  = genres['item'].map(lambda x : items_dict[x])
        director['item'] = director['item'].map(lambda x : items_dict[x])
        writer['item'] = writer['item'].map(lambda x : items_dict[x])
        writer_director['item'] = writer_director['item'].map(lambda x : items_dict[x])
        
    # year LabelEncoding
    le = LabelEncoder()
    year['year'] = le.fit_transform(year['year'])
    year = {i:v for i,v in zip(year['item'], le.fit_transform(year['year']))}
    # genre LabelEncoding
    genre_series = genres.sort_values(['item','genre']).groupby('item')['genre'].apply(list).apply(str) # pd.Series item,genres
    genre_le=le.fit_transform(genre_series.values)
    genres = {i:v for i,v in zip(genre_series.index, genre_le)}
    # director LabelEncoding
    # A,B,C 
    A_list = list(set(director['item'].values))  # 5503
    B_list = list(set(writer_director['item'].values) - set(director['item'].values) )  # 675
    C_list = list(set(rating_origin['item'].values) - set(writer_director['item'].values)) # 629

    director_group = director.groupby('item')['director'].apply(list)
    director_group = director_group.apply(lambda x:x[0]).to_dict()
    writer_group = writer.groupby('item')['writer'].apply(list)
    writer_group = writer_group.apply(lambda x:x[0]).to_dict()
    for i in B_list:
        director_group[i] = writer_group[i]
    for i in C_list:
        director_group[i] = 'no_direct'    

    director = {i:v for i,v in zip(director_group.keys(), le.fit_transform(list(director_group.values())))}  
    
    ########################################

    model_dir = args.output_dir
    model_name = args.model+".pt"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_all = args.train_all
    if args.use_all:
        dataset = DeepFMDataset_renew(rating_origin, year, genres,director,  valid_size=10, mode = 'seq', train_all = train_all)
        n_user,n_item,n_year,n_genre,n_director = dataset.get_num_context()
        if args.train_all:
            input_dims = [n_user,n_item,n_year,n_genre,n_director]
        else: 
            input_dims = [n_user, n_year,n_genre,n_director]

        model = DeepFM_renew(input_dims, args.embedding_dim, mlp_dims = args.mlp_dims).to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
    else:
        dataset = DeepFMDataset(rating_origin, year, genres, valid_size=10, mode = 'seq')
        n_user,n_item,n_year = dataset.get_num_context()
        input_dims = [n_user,n_item,n_year]

        model = DeepFM(input_dims, args.embedding_dim, mlp_dims = args.mlp_dims).to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))

    print('model setting done!')

    ########################################
    def make_joined_vector(user, items):
        user = torch.tensor([[user] for _ in range(len(items))])
        year_ = torch.tensor([[year[i]] for i in items])
        items_ = torch.tensor(items).unsqueeze(1)
        director_ = torch.tensor([[director[i]] for i in items])
        genre_ = torch.tensor([[genres[i]] for i in items])
        if train_all:
            return user.cuda(), items_.cuda(), torch.cat([user,items_,year_,genre_,director_], axis = 1).type(torch.long).cuda()
        return user.cuda(), items_.cuda(), torch.cat([user,year_,genre_,director_], axis = 1).type(torch.long).cuda()
    print('dataset setting done!')

    ##### inference #####
    # user, item set 
    users_set = set(rating_origin['user'].values)
    items_set = set(rating_origin['item'].values)
    submission = pd.DataFrame(columns=['user','item'])
    
    print("inference by user Start!")
    for user in tqdm(users_set):
        if args.rerank: # rerank 시에는 데이터에 있는 아이템들만 예측
            pred_items = rating_origin[rating_origin['user']==user]['item'].values
        else: # rerank 안할 시에는 본 아이템 제외 모든 아이템들 예측
            user_items_set = set(rating_origin[rating_origin['user']==user]['item'].values)
            pred_items = list(set(items_set).difference(user_items_set)) 
            
        tmp_user, tmp_item, x = make_joined_vector(user, pred_items) 
        model.eval()
        output = model(x)

        output = torch.cat([tmp_user, tmp_item, output.view(output.size(0),1)] , axis = 1).detach().cpu().numpy()
        output = pd.DataFrame(output, columns=['user','item', 'y'])
        output = output.sort_values(by='y', ascending=False)[:10]
        output.drop(columns='y',inplace=True)
        submission = pd.concat([submission,output])

    # Label 인코딩된 user, item을 원상복귀
    reverse_users= dict(map(reversed,users_dict.items()))
    reverse_items= dict(map(reversed,items_dict.items()))
    
    # submission 생성
    submission['user']  = submission['user'].map(lambda x : reverse_users[x])
    submission['item']  = submission['item'].map(lambda x : reverse_items[x])
    # submission_name = 'to_see.csv'
    submission.to_csv(os.path.join(args.output_dir,submission_name),index=False)
    
if __name__ == "__main__":
    main()
