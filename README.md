# level2-movie-recommendation-recsys-10
<img width="2562" alt="메인로고" src="https://user-images.githubusercontent.com/44939208/165333991-73456f44-3093-4395-aeac-5fd4e96d8a79.png">

## ❗ 주제 및 데이터 설명

- 사용자의 영화 평가 이력 데이터(user-item interaction)와 영화들의 다양한 정보(side-information)를 제공받음
- 해당 데이터를 기반으로 사용자가 다음에 시청하거나 이전에 시청했던 영화를 예측하는 추천 시스템을 개발


## 👋 팀원 소개

|                                                  [신민철](https://github.com/minchoul2)                                                   |                                                                          [유승태](https://github.com/yst3147)                                                                           |                                                 [이동석](https://github.com/dongseoklee1541)                                                  |                                                                        [이아현](https://github.com/ahyeon0508)                                                                         |                                                                         [임경태](https://github.com/gangtaro)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/52911772?v=4)](https://github.com/minchoul2) | [![Avatar](https://avatars.githubusercontent.com/u/39907669?v=4)](https://github.com/yst3147) | [![Avatar](https://avatars.githubusercontent.com/u/41297473?v=4)](https://github.com/dongseoklee1541) | [![Avatar](https://avatars.githubusercontent.com/u/44939208?v=4)](https://github.com/ahyeon0508) | [![Avatar](https://avatars.githubusercontent.com/u/45648852?v=4)](https://github.com/gangtaro) |

## ⚙ 프로젝트 수행 절차 및 방법
<img width="1676" alt="구조도" src="https://user-images.githubusercontent.com/44939208/165338548-3cb34143-0fd1-4d99-8686-e7bd7f0609c3.png">

- EDA | 주어진 데이터에 대한 파악   
- Data Processing | 주어진 데이터를 Model에 학습시킬 수 있도록 구조화 
- Modeling | 목적에 부합하는 모델 선정 및 개발  
- Training | 개발한 모델을 준비한 데이터로 학습  
- Validation | 학습된 모델을 평가하여 성능을 확인  
- Inference | 학습된 모델로 각 유저 당 10개의 아이템을 추천

## 🔨 Installation
- numpy==1.22.2
- pandas==1.4.1
- pytz==2021.3
- python-dateutil==2.8.2
- scipy==1.8.0
- six==1.16.0
- torch==1.10.2
- tqdm==4.62.3
- typing_extensions==4.1.1
- Python==3.8.5
- RecBole==1.0.1
- implicit==0.5.2
- catboost==1.0.5

## 🏢 Structure
```bash
level2-movie-recommendation-recsys-10
├── README.md
├── input
    ├── AutoRec
    │   ├── README.md
    │   ├── autorec.py
    │   ├── datasets.py
    │   ├── models.py
    │   ├── modules.py
    │   ├── trainers.py
    │   └── utils.py
    ├── Implicit_lib_model
    │   ├── README.md
    │   ├── implicit_model.py
    │   ├── models.py
    │   ├── modules.py
    │   └── utils.py
    ├── MF
    │   ├── README.md
    │   ├── mf.py
    │   ├── models.py
    │   ├── modules.py
    │   └── utils.py
    ├── MultiDAE
    │   ├── MultiDAE.py
    │   ├── README.md
    │   ├── datasets.py
    │   ├── models.py
    │   ├── preprocessing.py
    │   └── utils.py
    ├── MultiVAE
    │   ├── MultiVAE.py
    │   ├── README.md
    │   ├── datasets.py
    │   ├── models.py
    │   ├── preprocessing.py
    │   └── utils.py
    ├── NCF
    │   ├── README.md
    │   ├── datasets.py
    │   ├── models.py
    │   ├── modules.py
    │   ├── ncf.py
    │   ├── trainers.py
    │   └── utils.py
    ├── RecBole
    │   ├── ...
    ├── Transformer
    │   ├── README.md
    │   ├── datasets.py
    │   ├── inference.py
    │   ├── models.py
    │   ├── modules.py
    │   ├── preprocessing.py
    │   ├── run_pretrain.py
    │   ├── run_train.py
    │   ├── trainers.py
    │   └── utils.py
    ├── WideDeep
    │   ├── README.md
    │   ├── datasets.py
    │   ├── inference.py
    │   ├── models.py
    │   ├── modules.py
    │   ├── preprocessing.py
    │   ├── run_pretrain.py
    │   ├── run_train.py
    │   ├── trainers.py
    │   ├── utils.py
    │   └── wide_deep.ipynb
    ├── catboost
    │   ├── README.md
    │   ├── catboost.ipynb
    │   ├── catboost_inference.py
    │   └── run_catboost.py
    ├── code
    │   └── wide_deep.ipynb
    ├── deepfm
    │   ├── README.md
    │   ├── dataset.py
    │   ├── deepfm.py
    │   ├── inference.py
    │   ├── preprocessing.py
    │   ├── train.py
    │   └── utils.py
    ├── ease
    │   ├── __init__.py
    │   ├── ease-run.ipynb
    │   ├── metrics.py
    │   ├── model.py
    │   ├── readme.md
    │   └── train.py
    ├── gnn
    │   ├── dataset.py
    │   ├── inference.py
    │   └── models
    │   │   ├── __init__.py
    │   │   ├── lightgcn.py
    │   │   └── ngcf.py
    │   ├── negative_sampling_gnn.ipynb
    │   ├── readme.md
    │   ├── train.py
    │   ├── train_all.py
    │   ├── trainers.py
    │   └── utils.py
    ├── slim
    │   ├── __init__.py
    │   ├── datasets.py
    │   ├── hyper-parameter-tuning.ipynb
    │   ├── inference.py
    │   ├── models.py
    │   ├── readme.md
    │   └── slim-run.ipynb
    └── vae
        ├── dataset.py
        ├── inference.py
        ├── models
        │   ├── __init__.py
        │   ├── recvae.py
        │   └── recvae_ract.py
        ├── readme.md
        ├── train.py
        ├── train_all.py
        ├── train_ract.py
        └── utils.py
```

## ⚙️ Training 명령어
각 모델의 README를 참고하시면 됩니다 :)


## 📜 참고자료

[Embarrassingly Shallow Autoencoders for Sparse Data](https://arxiv.org/abs/1905.03375)

[RecVAE: a New Variational Autoencoder for Top-N Recommendations with Implicit Feedback](https://arxiv.org/abs/1912.11160)

[Variational Autoencoders for Collaborative Filtering](https://arxiv.org/pdf/1802.05814.pdf)

[AutoRec: Autoencoders Meet Collaborative Filtering](https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)

[Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)

[DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)

[ADMM SLIM: Sparse Recommendations for Many Users](http://www.cs.columbia.edu/~jebara/papers/wsdm20_ADMM.pdf)

[Implicit](https://github.com/benfred/implicit)

[RecBole](https://github.com/RUCAIBox/RecBole)

