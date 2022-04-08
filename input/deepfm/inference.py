import os
import argparse

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

from models.deepfm import DeepFM

def main():
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default='42', help=' ')
    
    parser.add_argument('--v', type=str, default='1')
    parser.add_argument('--data_dir', type=str, default='../data/train')
    parser.add_argument("--output_dir", default="../../output/", type=str)
    parser.add_argument('--mlp_dims', nargs='+', type=int, default=[30,20,10], help = 'Multi-Layer-Perceptron dimensions list')
    parser.add_argument('--embedding_dim', type=int, default=10, help='embedding_dim for input tensor')

    args = parser.parse_args()

    # 모델 불러오기
    model_dir = args.output_dir
    model_name = "DeepFM_1_loss0.3467.pt"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    input_dims = [31360, 6807, 93]

    model = DeepFM(input_dims, args.embedding_dim, mlp_dims=[30, 20, 10]).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
    print('model setting done!')

    ##### 데이터 셋 구성 #####
    data_dir = "../data/train"
    # 유저, 아이템, year,genre index 인코딩
    rating_df = pd.read_csv(os.path.join(data_dir, 'train_ratings.csv'))   
    genres_df = pd.read_csv(os.path.join(data_dir, 'genres.tsv'), sep='\t') 
    years_df = pd.read_csv(os.path.join(data_dir,'years.tsv'), sep='\t')

    users = list(set(rating_df.loc[:,'user']))
    users.sort()
    items =  list(set((rating_df.loc[:, 'item'])))
    items.sort()
    
    if len(users)-1 != max(users):
        users_dict = {users[i]: i for i in range(len(users))}
        rating_df['user']  = rating_df['user'].map(lambda x : users_dict[x])

    if len(items)-1 != max(items):
        items_dict = {items[i]: i for i in range(len(items))}
        rating_df['item']  = rating_df['item'].map(lambda x : items_dict[x])
        years_df['item']  = years_df['item'].map(lambda x : items_dict[x])
        genres_df['item']  = genres_df['item'].map(lambda x : items_dict[x])

    le = LabelEncoder()
    mlb = MultiLabelBinarizer()
    years_df['year'] = le.fit_transform(years_df['year'])

    # genre Multi Label Binarize 
    genre_dict = {genre:i for i, genre in enumerate(set(genres_df['genre']))}
    genres_df['genre'] = genres_df['genre'].map(lambda x : genre_dict[x])
    genres_df = genres_df.groupby('item')['genre'].agg(lambda x : list(x))
    multi_genres = mlb.fit_transform(genres_df.values)
    genres_df = pd.DataFrame([genres_df.index, multi_genres]).T.rename({0:'item',1:'genre'},axis =1)

    # 영화 데이터에 장르, 연도 합치기
    def make_joined_vector(user, items):
        user = torch.tensor([user for _ in range(len(items))]).unsqueeze(1)
        genre = torch.from_numpy(np.vstack(genres_df.loc[items]['genre'].values).astype(float))
        year = torch.tensor(years_df.loc[items]['year'].values).unsqueeze(1)
        items = torch.tensor(items).unsqueeze(1)
        return user, items, torch.cat([user,items,year,genre], axis = 1), 
    print('dataset setting done!')

    ##### inference #####
    users = set(rating_df['user'].values)
    items = set(rating_df['item'].values)
    submission = pd.DataFrame(columns=['user','item'])

    for user in tqdm(users):
        user_items_set = set(rating_df[rating_df['user']==user]['item'].values)
        pred_items = list(set(items).difference(user_items_set)) 
        tmp_user, tmp_item, x = make_joined_vector(user, pred_items) 

        model.eval()
        output = model(x)
        
        output = torch.cat([tmp_user, tmp_item, output.view(output.size(0),1)] , axis = 1)
        output = pd.DataFrame(output, columns=['user','item', 'y'])
        output = output.sort_values(by=2, ascending=False)[:10]
        output.drop(columns='y',inplace=True)
        submission = pd.concat([submission,output])

    reverse_users= dict(map(reversed,users.items()))
    reverse_items= dict(map(reversed,items.items()))
    
    submission['user']  = submission['user'].map(lambda x : reverse_users[x])
    submission['item']  = submission['item'].map(lambda x : reverse_items[x])
    submission.to_csv('../../output/deepfm_submission.csv',index=False)
    
if __name__ == "__main__":
    main()
