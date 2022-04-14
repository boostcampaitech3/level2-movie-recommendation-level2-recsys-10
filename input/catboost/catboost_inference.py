import os
import argparse
from re import sub
from xmlrpc.client import boolean

import pandas as pd
import numpy as np
from tqdm import tqdm

from catboost import CatBoostClassifier
from sklearn.preprocessing import MultiLabelBinarizer

def main():
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default='42', help=' ')
    parser.add_argument('--rerank', type=boolean, default=False, help='IF True: re-rank top 50 per user to top10')

    parser.add_argument('--data_dir', type=str, default='../data/train')
    parser.add_argument("--output_dir", default="./output/", type=str)
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--candidate", type=str, default='submission-RecVAE-top50.csv')

    args = parser.parse_args()

    ############################################
    ##### 데이터 셋 구성 전처리 #####
    data_dir = "../data/train"
    genre_df = pd.read_csv(os.path.join(data_dir, 'genres.tsv'), sep='\t') 
    print("Dataset setting") 
    rating = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'))


    # 유저, 아이템 Encoding
    submission_name = args.model + '_all.csv'
    if args.rerank:
        candidate_file = args.candidate
        rating = pd.read_csv(os.path.join(data_dir, candidate_file))
        submission_name = args.model + 'submission.csv'
        print(f'##Rerank-Mode by {candidate_file} to {submission_name}##')
    else:
        rating = rating.drop('time', axis=1)

    # genre LabelEncoding
    genre_data = genre_df.copy()
    genre_dict = {genre:i for i , genre in enumerate(set(genre_df['genre']))}
    genre_data['genre'] = genre_data['genre'].map(lambda x : genre_dict[x])
    sum_genre = list()
    for item in genre_data['item'].unique():
        sum_genre.append([item, genre_data[genre_data['item']==item]['genre'].values])
    sum_genre = pd.DataFrame(sum_genre, columns=['item', 'genre'])
    mlb = MultiLabelBinarizer()
    genre_label = mlb.fit_transform(sum_genre['genre'])
    sum_genre = pd.concat([sum_genre['item'], pd.DataFrame(genre_label, columns = genre_dict)], axis=1)

    ########################################
    model_dir = args.output_dir
    model_name = args.model+".bin"

    model = CatBoostClassifier()
    model.load_model(os.path.join(model_dir, model_name))
    print('model setting done! : ', model.get_params())
    ########################################
    

    ##### inference #####
    # user, item set 
    users_set = set(rating['user'].values)
    items_set = set(rating['item'].values)
    submission = pd.DataFrame(columns=['user','item'])
    
    print("inference by user Start!")
    if args.rerank: # rerank 시에는 데이터에 있는 아이템들만 예측
        submission_genre = pd.merge(rating, sum_genre, on='item')
        

    else: # rerank 안할 시에는 본 아이템 제외 모든 아이템들 예측
        submission_genre = pd.DataFrame(columns=['user','item'])
        for user in tqdm(users_set):
            user_items_set = set(rating[rating['user']==user]['item'].values)
            pred_items = list(set(items_set).difference(user_items_set))
            catcu  = pd.DataFrame({'user' : [user] * len(pred_items) , 'item' : pred_items})
            submission_genre = pd.concat([submission_genre, catcu])
        submission_genre = pd.merge(submission_genre , sum_genre, on='item')
    
    submission_genre['y'] = 0
    preds_proba = model.predict_proba(submission_genre)
    submission_genre['y'] = preds_proba[:,1]
    submission_genre = submission_genre[['user','item','y']]
    
    remove_sub = pd.DataFrame(columns=['user', 'item'])
    for user in tqdm(users_set):
        output = submission_genre[submission_genre['user']==user].sort_values(by='y', ascending=False)[:10]
        output.drop(columns='y' , inplace=True)
        remove_sub = pd.concat([remove_sub, output])
    
    # # 장르정보 추가
    # x = pd.merge(rating, sum_genre, on='item')
        
    #     # 각 row 별 확률값 반환
    #     preds_proba = model.predict_proba(x)
    #     x['y'] = preds_proba[:,1]

    #     output = pd.DataFrame(x, columns=['user','item', 'y'])
    #     output = output.sort_values(by='y', ascending=False)[:10]
    #     output.drop(columns='y',inplace=True)
    #     submission = pd.concat([submission,output])

    # submission 생성
    # submission['user']  = submission['user'].map(lambda x : reverse_users[x])
    # submission['item']  = submission['item'].map(lambda x : reverse_items[x])
    # submission_name = 'to_see.csv'
    remove_sub.sort_values(by='user', ascending=True, inplace=True)
    remove_sub.to_csv(os.path.join(args.output_dir,submission_name),index=False)


if __name__ == "__main__":
    main()