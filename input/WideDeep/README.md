# Movie Recommendation Baseline Code

영화 추천 대회를 위한 WideDeep 코드입니다. inference 까지는 제작하지 못했지만, Datasets과 Model 구조를 만들었습니다.

## How to run

1. Pretraining
   ```
   python run_pretrain.py
   ```
2. Fine Tuning (Main Training)
   1. with pretrained weight
      ```
      python run_train.py --using_pretrain
      ```
   2. without pretrained weight
      ```
      python run_train.py
      ```
3. Inference
   ```
   python inference.py
   ```


- 코드 출처: https://github.com/aHuiWang/CIKM2020-S3Rec
