import argparse
import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

from utils import neg_sample_popular

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepfm", help="")
    args = parser.parse_args()
    
    data_dir = "../data/train" 

    print(f'{args.model} Preprocessing start...')
    # 모델별 전처리
    if args.model == "deepfm":
        # rating df
        train_rating = os.path.join(data_dir, 'train_ratings.csv')
        rating_df = pd.read_csv(train_rating)
        
        rating_df['rating'] = 1.0 # implicit feedback
        users = set(rating_df.loc[:, 'user'])
        items = set(rating_df.loc[:, 'item'])
        print("1. rating done")

        # Genre df 
        genre = os.path.join(data_dir, 'genres.tsv')
        genres_df = pd.read_csv(genre, sep='\t')
        genre_dict = {genre:i for i, genre in enumerate(set(genres_df['genre']))}
        genres_df['genre'] = genres_df['genre'].map(lambda x : genre_dict[x])
        genres_df = genres_df.groupby('item')['genre'].agg(lambda x : list(x))

        mlb = MultiLabelBinarizer()
        multi_genres = mlb.fit_transform(genres_df.values)
        genres_df = pd.DataFrame([genres_df.index, multi_genres]).T.rename({0:'item',1:'genre'},axis =1)
        print("2. genre done")

        # Year df
        year = os.path.join(data_dir,'years.tsv')
        years_df = pd.read_csv(year, sep='\t')
        le = LabelEncoder()
        years_df['year'] = le.fit_transform(years_df['year'])
        print("3. year done")

        # Negative sample by Popularity
        popular_items = list(rating_df.groupby('item').count()['user'].sort_values(ascending=False).index) # 인기도순 아이템 리스트
        
        user_neg_dict = dict()
        for user in tqdm(users):
            popular_rate = 0.4
            item_set = set(rating_df[rating_df['user']==user]['item']) 
            if len(item_set)*2 > len(popular_items)*popular_rate: # 본 영화가 너무 많으면 전체에서 neg_sample
                popular_rate = 1
            user_neg_dict[user] = neg_sample_popular(popular_item_list=popular_items, item_set=item_set, 
                                                    neg_sample_by_pos_sample=True, popular_rate=popular_rate)
        u_neg_list = []
        i_neg_list = []
        for user,items in user_neg_dict.items():
            u_neg_list.extend([user]*len(items))
            i_neg_list.extend([*items])
        
        user_neg_df = pd.DataFrame({'user':u_neg_list,'item':i_neg_list, 'rating':0})
        rating_df = pd.concat([rating_df, user_neg_df], axis = 0)
        print("4. NegativeSampling done")

        # Join dfs
        rating_df = pd.merge(rating_df, genres_df, how = 'inner', on = 'item')
        rating_df = pd.merge(rating_df, years_df, how = 'inner', on= 'item')
        print("5. Join dataframe done")

        # zero-based index로 mapping
        users = list(set(rating_df.loc[:,'user']))
        users.sort()
        items =  list(set((rating_df.loc[:, 'item'])))
        items.sort()

        if len(users)-1 != max(users):
            users_dict = {users[i]: i for i in range(len(users))}
            rating_df['user']  = rating_df['user'].map(lambda x : users_dict[x])
            users = list(set(rating_df.loc[:,'user']))
            
        if len(items)-1 != max(items):
            items_dict = {items[i]: i for i in range(len(items))}
            rating_df['item']  = rating_df['item'].map(lambda x : items_dict[x])
            items =  list(set((rating_df.loc[:, 'item'])))

        rating_df = rating_df.sort_values(by=['user'])
        rating_df.reset_index(drop=True, inplace=True)
        print("6. zero-based index done")

        # to_data 
        output_path = os.path.join(data_dir,'deepfm_train.json')
        rating_df.drop(columns='time',inplace=True)
        rating_df.to_json(output_path)
        
        print(f'{args.model} Preprocessing done!!!')
        print(f'the file located in {output_path}')

    else:
        pass

if __name__ == "__main__":
    main()
