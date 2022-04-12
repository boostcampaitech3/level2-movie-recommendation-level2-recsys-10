# @Time   : 2021/03/20
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn


"""
Case study example
===================
Here is the sample code for the case study in RecBole.
"""


import torch
from recbole.utils.case_study import full_sort_topk, full_sort_scores
from recbole.quick_start import load_data_and_model

import os
import pandas as pd

if __name__ == '__main__':
    train_df = pd.read_csv("../recbole/dataset_example/ml-100k/train_ratings.csv")
    user_list = train_df['user'].unique().astype(str)

    for model_path in os.listdir('../saved'):
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file = os.path.join('../saved/') + model_path
        )  # Here you can replace it by your model path.

        # uid_series = np.array([1, 2])  # internal user id series
        # or you can use dataset.token2id to transfer external user token to internal user id

        uid_series = dataset.token2id(dataset.uid_field, user_list)

        topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config['device'])
        print(topk_score)  # scores of top 10 items
        print(topk_iid_list)  # internal id of top 10 items
        external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
        print(external_item_list)  # external tokens of top 10 items
        print()
        
        result_df = pd.DataFrame(external_item_list, index=train_df['user'].unique()).stack().reset_index()

        result_df.rename(columns={'level_0': 'user', 0: 'item'}, inplace=True)
        result_df.drop(['level_1'], axis=1, inplace=True)

        output_path = "./output"
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        
        result_df.to_csv("output/" + model_path[:-4] + ".csv", index=False)

        score = full_sort_scores(uid_series, model, test_data, device=config['device'])
        print(score)  # score of all items

        print("------ " + model_path[:-4] + " inference complete ------")
        print("========================================================")