# NCF

Movie Recommendation NCF(Neural Collaborative Filtering) Code

- [paper](https://arxiv.org/abs/1708.05031)
- [model reference code](https://github.com/yst3147/Recsys_Implement/blob/main/NCF/NCF.ipynb)

   BoostCamp AI Tech RecSys Level 2 U-stage 기본과제 2 참고


## How to run

> ### NCF Multi-Layer Perceptron
  
   ```
   # without wandb
   python ncf.py --model_name NCF

   # with wandb
   python ncf.py --model_name NCF --wandb
   ```

> ### NCF Generalized Matrix Factorization (GMF)

   ```
   # without wandb
   python ncf.py --model_name GMF

   # with wandb
   python ncf.py --model_name GMF --wandb
   ```

> ### NCF Fusion of GMF and MLP (NeuMF)

   ```
   # without wandb
   python ncf.py --model_name NeuMF

   # with wandb
   python ncf.py --model_name NeuMF --wandb
   ```

