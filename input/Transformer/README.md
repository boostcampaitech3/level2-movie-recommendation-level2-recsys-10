# Transformer

Movie Recommendation with Transformer Base Model

## Model
> ### S3Rec (Pretrain Model)
- [paper](https://arxiv.org/abs/2008.07873v1)
- [model reference code](https://github.com/aHuiWang/CIKM2020-S3Rec)

> ### SASRec
- [paper](https://arxiv.org/abs/1808.09781v1)
- [model reference code](https://github.com/aHuiWang/CIKM2020-S3Rec)

    BoostCamp AI Tech RecSys Level 2 P-stage Special Mission 4 참고

> ### BERT4Rec
- [paper](https://arxiv.org/abs/1904.06690v2)
- [model reference code](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch)

    BoostCamp AI Tech RecSys Level 2 P-stage Special Mission 5 참고

## How to run

> ### SASRec
1. Pretraining with S3Rec
   ```
   python run_pretrain.py
   ```
2. Fine Tuning (Main Training)
   1. with pretrained weight
      ```
      python run_train.py --model_name SASRec --using_pretrain
      ```
   2. without pretrained weight
      ```
      python run_train.py --model_name SASRec 
      ```

3. Inference
   ```
   python inference.py --model_name SASRec 
   ```

> ### Bert4Rec

1. Main Training
   ```
   python run_train.py --model_name BERT4Rec 
   ```

2. Inference
   ```
   python inference.py --model_name BERT4Rec 
   ```