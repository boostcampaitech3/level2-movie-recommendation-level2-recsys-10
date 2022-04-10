import os
import argparse
from xmlrpc.client import boolean

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
    parser.add_argument('--rerank', type=boolean, default=False, help='IF True: re-rank top 50 per user to top10')

    parser.add_argument('--data_dir', type=str, default='../data/train')
    parser.add_argument("--output_dir", default="./output/", type=str)
    parser.add_argument('--mlp_dims', nargs='+', type=int, default=[30,20,10], help = 'Multi-Layer-Perceptron dimensions list')
    parser.add_argument('--embedding_dim', type=int, default=10, help='embedding_dim for input tensor')

    args = parser.parse_args()

    # 모델 불러오기
    model_dir = args.output_dir
    model_name = "DeepFM_3.pt"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    input_dims = [31360, 6807, 101]

    model = DeepFM(input_dims, args.embedding_dim, mlp_dims=[30, 20, 10]).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
    print('model setting done!')

    ##### 데이터 셋 구성 #####
    data_dir = "../data/train"
    # 유저, 아이템, year,genre index 인코딩
    rating_df = pd.read_csv(os.path.join(data_dir, 'train_ratings.csv'))   
    genres_df = pd.read_csv(os.path.join(data_dir, 'genres.tsv'), sep='\t') 
    years_df = pd.read_csv(os.path.join(data_dir,'years.tsv'), sep='\t')
    
    # year 결측치 처리
    not_year = list(set(rating_df['item']) - set(years_df['item']))
    not_year.sort()
    null_year = [1921,1920,1919,1915,1916,1917,1902,2015]
    missing_year = pd.DataFrame({'item':not_year, 'year':null_year})
    years_df = pd.concat([years_df, missing_year])

    # user, item 인코딩을 위한 리스트
    users = list(set(rating_df.loc[:,'user']))
    users.sort()
    items =  list(set((rating_df.loc[:, 'item'])))
    items.sort()

    output_dif = '../../output/'
    submission_name = 'deepfm_submission.csv'

    # rerank시 rating_df / submission 이름 변경 
    if args.rerank:
        candidate_file = 'submission-RecVAE-top50.csv'
        rating_df = pd.read_csv(os.path.join(data_dir, candidate_file))
        submission_name = 'deepfm_submission_rerank.csv'
        print(f'##Rerank-Mode by {candidate_file} to {submission_name}##')

    # User, Item Label 인코딩    
    if len(users)-1 != max(users):
        users_dict = {users[i]: i for i in range(len(users))}
        rating_df['user']  = rating_df['user'].map(lambda x : users_dict[x])

    if len(items)-1 != max(items):
        items_dict = {items[i]: i for i in range(len(items))}
        rating_df['item']  = rating_df['item'].map(lambda x : items_dict[x])
        years_df['item']  = years_df['item'].map(lambda x : items_dict[x])
        genres_df['item']  = genres_df['item'].map(lambda x : items_dict[x])
    
    # Year Label 인코딩
    le = LabelEncoder()
    mlb = MultiLabelBinarizer()
    years_df['year'] = le.fit_transform(years_df['year'])
    years_df = years_df.sort_values(by='item').reset_index().drop(columns='index')

    # genre Multi Label Binarize 
    genre_dict = {genre:i for i, genre in enumerate(set(genres_df['genre']))}
    genres_df['genre'] = genres_df['genre'].map(lambda x : genre_dict[x])
    genres_df = genres_df.groupby('item')['genre'].agg(lambda x : list(x))
    multi_genres = mlb.fit_transform(genres_df.values)
    genres_df = pd.DataFrame([genres_df.index, multi_genres]).T.rename({0:'item',1:'genre'},axis =1)

    # 영화 데이터에 장르, 연도 합치는 함수
    def make_joined_vector(user, items):
        user = torch.tensor([user for _ in range(len(items))]).unsqueeze(1)
        genre = torch.from_numpy(np.vstack(genres_df.loc[items]['genre'].values).astype(float))
        year = torch.tensor(years_df.loc[items]['year'].values).unsqueeze(1)
        items = torch.tensor(items).unsqueeze(1)
        return user.cuda(), items.cuda(), torch.cat([user,items,year,genre], axis = 1).type(torch.long).cuda(), 
    print('dataset setting done!')

    ##### inference #####
    # user, item set 
    users_set = set(rating_df['user'].values)
    items_set = set(rating_df['item'].values)
    submission = pd.DataFrame(columns=['user','item'])
    
    print("inference by user Start!")
    for user in tqdm(users_set):
        if args.rerank: # rerank 시에는 데이터에 있는 아이템들만 예측
            pred_items = rating_df[rating_df['user']==user]['item'].values
        else: # rerank 안할 시에는 본 아이템 제외 모든 아이템들 예측
            user_items_set = set(rating_df[rating_df['user']==user]['item'].values)
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
