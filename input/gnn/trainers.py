import torch
from utils import split_matrix, compute_ndcg_k


def train(model, data_generator, optimizer):
    """
    Train the model PyTorch style

    Arguments:
    ---------
    model: PyTorch model
    data_generator: Data object
    optimizer: PyTorch optimizer
    """
    model.train()
    n_batch = data_generator.n_train // data_generator.batch_size + 1
    running_loss=0
    for _ in range(n_batch):
        u, i, j = data_generator.sample()
        optimizer.zero_grad()
        loss = model(u,i,j)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss


def eval_model(u_emb, i_emb, Rtr, Rte, k, device):
    """
    Evaluate the model
    
    Arguments:
    ---------
    u_emb: User embeddings
    i_emb: Item embeddings
    Rtr: Sparse matrix with the training interactions
    Rte: Sparse matrix with the testing interactions
    k : kth-order for metrics
    
    Returns:
    --------
    result: Dictionary with lists correponding to the metrics at order k for k in Ks
    """
    # split matrices
    ue_splits = split_matrix(u_emb)
    tr_splits = split_matrix(Rtr)
    te_splits = split_matrix(Rte)

    recall_k, ndcg_k= [], []
    # compute results for split matrices
    for ue_f, tr_f, te_f in zip(ue_splits, tr_splits, te_splits):

        scores = torch.mm(ue_f, i_emb.t())

        test_items = torch.from_numpy(te_f.todense()).float().to(device)
        non_train_items = torch.from_numpy(1-(tr_f.todense())).float().to(device)
        scores = scores * non_train_items

        _, test_indices = torch.topk(scores, dim=1, k=k)
        

        pred_items = torch.zeros_like(scores).float()
        pred_items.scatter_(dim=1, index=test_indices, src=torch.ones_like(test_indices).float().to(device))

        topk_preds = torch.zeros_like(scores).float()
        topk_preds.scatter_(dim=1, index=test_indices[:, :k], src=torch.ones_like(test_indices).float())
        
        TP = (test_items * topk_preds).sum(1)                      
        rec = TP/test_items.sum(1)
   
        ndcg = compute_ndcg_k(pred_items, test_items, test_indices, k, device)

        recall_k.append(rec)
        ndcg_k.append(ndcg)

    return torch.cat(recall_k).mean(), torch.cat(ndcg_k).mean()
