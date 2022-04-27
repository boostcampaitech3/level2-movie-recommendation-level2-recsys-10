# Implicit Library Model

Movie Recommendation with Implicit Library

- [reference code](https://github.com/benfred/implicit)

## Installation

```
pip install implicit
```

## model
 * Alternating Least Squares as described in the papers [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) and [Applications of the Conjugate Gradient Method for Implicit
Feedback Collaborative Filtering](https://pdfs.semanticscholar.org/bfdf/7af6cf7fd7bb5e6b6db5bbd91be11597eaf0.pdf).

 * [Bayesian Personalized Ranking](https://arxiv.org/pdf/1205.2618.pdf).

 * [Logistic Matrix Factorization](https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)

 * Item-Item Nearest Neighbour models using Cosine, TFIDF or BM25 as a distance metric.


## How to run

> ### Alternating Least Squares
  
   ```
   # without bm25 weight
   python implicit_model.py --model_name ALS --hidden_size 100 --regularization 0.01 --iterations 15

   # with bm25 weight
   python implicit_model.py --model_name ALS --hidden_size 100 --regularization 0.01 --iterations 15 --bm25 --bm25_B 0.9
   ```

> ### Bayesian Personalized Ranking
   ```
   python implicit_model.py --model_name BPR --hidden_size 100 --regularization 0.01 --iterations 100 --lr 0.01
   ```

> ### Logistic Matrix Factorization
   ```
   python implicit_model.py --model_name LMF --hidden_size 30 --regularization 0.6 --iterations 30 --lr 1.00
   ```

> ### Item-Item Nearest Neighbour models using TFIDF distance metric
   ```
   python implicit_model.py --model_name TFIDF 
   ```

> ### Item-Item Nearest Neighbour models using COSINE distance metric
   ```
   python implicit_model.py --model_name COSINE
   ```

> ### Item-Item Nearest Neighbour models using BM25 distance metric
   ```
   python implicit_model.py --model_name BM25 --bm25_B 0.2
   ```
