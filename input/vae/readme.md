## Run
### train.py
- only for RecVAE

### train_all.py
- only for RecVAE
- No Validation split, Just train all the data for make the best score.

### train_RaCT.py
- only for RecVAE with RaCT (But, it didn't work)


## Models

### RecVAE
- Adapted from: https://github.com/ilya-shenbin/RecVAE
- paper: https://arxiv.org/abs/1912.11160
- Author: Gyeongtae Im (igt0530@gmail.com)

### RaCT (Didn't work)
- Adapted from: https://github.com/samlobel/RaCT_CF, https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/ract.py (based only Multi-VAE)
- paper: https://arxiv.org/abs/1906.04281v2
- Author: Gyeongtae Im (igt0530@gmail.com)

Tried to apply RaCT to RecVAE, But it didn't work well ... 


## Datasets

### Basedataset
- you can get sparse interaction matrix from this.
- validation rule
    - each user, choose maximum 10 of user's items.
        - 6, totally ramdom choose item with out 4 last squence
        - 4, sequentially last 4 item
