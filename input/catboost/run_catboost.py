import argparse
import os
import wandb
import pandas as pd
import numpy as np

from tqdm import tqdm
from catboost import CatBoostClassifier

from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

def accuracy(preds_class, y_test):
    correct_result_sum = 0
    for i in range(0, len(preds_class)):
        if preds_class[i] == y_test['target'][i]:
            correct_result_sum += 1

    return correct_result_sum / len(preds_class)

def confusion(output, y):
    tp, fp = 0, 0
    fn ,tn = 0, 0
    precision = 0
    recall = 0
    for i in range(0,len(output)):
        if y['target'][i] == 0:
            if output[i] == 0:
                tn += 1
            else:
                fp += 1
        elif y['target'][i] == 1:
            if output[i] == 1:
                tp += 1
            else:
                fn += 1
    precision = tp / (tp+fp)
    recall = tp / (tp + fn)
    return precision , recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type" , type=str, default="GPU", help="CPU or GPU")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--iterations" , type=int, default="1000", help="catboost iterations")
    parser.add_argument("--seed", type=int, default="42", help="random seed")
    parser.add_argument("--loss", type=str, default="Logloss", help="Logloss")
    parser.add_argument("--learning_rate" , type=float, default ="0.035176", help="setting learning_Rate")
    parser.add_argument("--wandb_name", type=str, help="wandb name")

    args = parser.parse_args()
    wandb.init(project="movierec_train", entity="egsbj")
    wandb.run.name = args.wandb_name
    wandb.config.update(args)

    output_folder = "./output/"

    # data loader
    # train_df = pd.read_csv('../data/train/train_ratings.csv') # 전체 학습 데이터
    # genre_df = pd.read_csv(os.path.join('../data/', 'train/genres.tsv'), sep='\t')

    # dataset = data_loader(train_df, genre_df)

    train_data = pd.read_csv('../data/train/test_dataset35%10.csv')
    train_labels = train_data['target']
    del train_data['target']

    X_train, X_test, y_train, y_test = train_test_split(
    train_data, train_labels, test_size=0.3, random_state=42
    )

    cat_features = list(range(0, X_train.shape[1]))
    y_test = y_test.reset_index()
    del y_test['index']
    custom_loss = ["AUC" , "Accuracy", "Recall", "Precision"]
    list_run = ['learn', 'validation']
    print("training Catboost...")
    model = CatBoostClassifier(iterations=args.iterations,
                           task_type=args.task_type,
                           devices='0:1',
                           loss_function=args.loss,
                           random_seed=args.seed,
                           custom_loss = custom_loss
                           )
    fit_model = model.fit(X_train,
          y_train,
          cat_features = cat_features,
          eval_set =(X_test, y_test),
          verbose=True)

    eval_result = model.get_evals_result()
    
    print("predicting Catboost..")
    preds_class = fit_model.predict(X_test)
    #preds_proba = model.predict_proba(X_test)

    acc = accuracy(preds_class, y_test)
    precision , recall = confusion(preds_class, y_test)
    AUC_score = metrics.roc_auc_score(y_test, preds_class)

    print("acc : ", acc)
    print("precision : ", precision)
    print("recall : ", recall)
    print("AUC score : ", AUC_score)
    
    print("logging wandb...")
    for i in tqdm(range(0,args.iterations)):
        wandb.log({
            "train_Logloss" : eval_result[list_run[0]]['Logloss'][i],
            "train_Accuracy" : eval_result[list_run[0]][custom_loss[1]][i],
            "train_Recall" : eval_result[list_run[0]][custom_loss[2]][i],
            "train_precision " : eval_result[list_run[0]][custom_loss[3]][i],
            "eval_Logloss" : eval_result[list_run[1]]['Logloss'][i],
            "eval_Accuracy" : eval_result[list_run[1]][custom_loss[1]][i],
            "eval_Recall" : eval_result[list_run[1]][custom_loss[2]][i],
            "eval_precision " : eval_result[list_run[1]][custom_loss[3]][i],
            "eval_AUC score " : eval_result[list_run[1]][custom_loss[0]][i]
        })
    print("model saved..")
    model.save_model(os.path.join(output_folder, args.wandb_name) +'.bin')
    print("end of catboost..")
if __name__ == "__main__":
    main()


# def data_loader(train_df, genre_df):

#     return dataset

