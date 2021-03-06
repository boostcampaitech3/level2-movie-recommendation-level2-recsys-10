# ref-1. https://github.com/RUCAIBox/RecBole/blob/b7b29954fbc963a406ddbe2f917b4ba56bd7a22b/docs/source/user_guide/model/general/ngcf.rst
# ref-2. ...
# Model Hyper-Parameters: (Default)
# ---------------------------------
# emb_dim = 64
# layers = [64, 64, 64]
# node_dropout = 0
# mess_dropout = 0.1
# reg = 1e-5

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp

class NGCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, layers, reg, node_dropout, mess_dropout, adj_mtx, device, mode = 'train'):
        super().__init__()
        self.device = device

        # initialize Class attributes
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.l_matrix = adj_mtx
        self.l_plus_i_matrix = adj_mtx + sp.eye(adj_mtx.shape[0])
        self.reg = reg
        self.layers = layers
        self.n_layers = len(self.layers)
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout

        # Initialize weights
        self.weight_dict = self._init_weights()
        print("Weights initialized.")

        # Create Matrix 'L+I', PyTorch sparse tensor of SP adjacency_mtx
        self.L_plus_I = self._convert_sp_mat_to_sp_tensor(self.l_plus_i_matrix)
        self.L = self._convert_sp_mat_to_sp_tensor(self.l_matrix)

        # this is for load_state_dict 
        self.u_final_embeddings = nn.Parameter(torch.empty(self.n_users, self.emb_dim + sum(self.layers)).to(self.device))
        self.i_final_embeddings = nn.Parameter(torch.empty(self.n_items, self.emb_dim + sum(self.layers)).to(self.device))

    # initialize weights
    def _init_weights(self):
        print("Initializing weights...")
        weight_dict = nn.ParameterDict()

        initializer = torch.nn.init.xavier_uniform_
        
        weight_dict['user_embedding'] = nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim).to(self.device)))
        weight_dict['item_embedding'] = nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim).to(self.device)))

        weight_size_list = [self.emb_dim] + self.layers

        for k in range(self.n_layers):
            weight_dict['W_one_%d' %k] = nn.Parameter(initializer(torch.empty(weight_size_list[k], weight_size_list[k+1]).to(self.device)))
            weight_dict['b_one_%d' %k] = nn.Parameter(initializer(torch.empty(1, weight_size_list[k+1]).to(self.device)))
            
            weight_dict['W_two_%d' %k] = nn.Parameter(initializer(torch.empty(weight_size_list[k], weight_size_list[k+1]).to(self.device)))
            weight_dict['b_two_%d' %k] = nn.Parameter(initializer(torch.empty(1, weight_size_list[k+1]).to(self.device)))
           
        return weight_dict

    # convert sparse matrix into sparse PyTorch tensor
    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        Convert scipy sparse matrix to PyTorch sparse matrix

        Arguments:
        ----------
        X = Adjacency matrix, scipy sparse matrix
        """
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        res = torch.sparse.FloatTensor(i, v, coo.shape).to(self.device)
        return res

    # apply node_dropout
    def _droupout_sparse(self, X):
        """
        Drop individual locations in X
        
        Arguments:
        ---------
        X = adjacency matrix (PyTorch sparse tensor)
        dropout = fraction of nodes to drop
        noise_shape = number of non non-zero entries of X
        """
        node_dropout_mask = ((self.node_dropout) + torch.rand(X._nnz())).floor().bool().to(self.device)
        i = X.coalesce().indices()
        v = X.coalesce()._values()
        i[:,node_dropout_mask] = 0
        v[node_dropout_mask] = 0
        X_dropout = torch.sparse.FloatTensor(i, v, X.shape).to(X.device)

        return  X_dropout.mul(1/(1-self.node_dropout))

    def forward(self, u, i, j):
        """
        Computes the forward pass
        
        Arguments:
        ---------
        u = user
        i = positive item (user interacted with item)
        j = negative item (user did not interact with item)
        """
        # apply drop-out mask
        L_plus_I_hat = self._droupout_sparse(self.L_plus_I) if self.node_dropout > 0 else self.L_plus_I
        L_hat = self._droupout_sparse(self.L) if self.node_dropout > 0 else self.L
        
        # ?????? ?????? (1)
        ego_embeddings = torch.cat([self.weight_dict['user_embedding'], self.weight_dict['item_embedding']], 0)

        final_embeddings = [ego_embeddings]

        # forward pass for 'n' propagation layers
        for k in range(self.n_layers):
            
            ### ?????? ?????? (7) ###
            
            # (L+I)E
            side_L_plus_I_embeddings = torch.sparse.mm(L_plus_I_hat, ego_embeddings) #?????? : use torch.sparse.mm 
            
            # (L+I)EW_1 + b_1
            simple_embeddings = torch.matmul(side_L_plus_I_embeddings, self.weight_dict['W_one_%d' %k]) + self.weight_dict['b_one_%d' % k] #?????? : use torch.matmul, self.weight_dict['W_one_%d' % k], self.weight_dict['b_one_%d' % k]
            
            # LE
            side_L_embeddings = torch.sparse.mm(L_hat, ego_embeddings) #?????? : use torch.sparse.mm                                
            
            # LEE
            interaction_embeddings = torch.mul(side_L_embeddings, ego_embeddings)            #?????? : use torch.mul
                                             
            # LEEW_2 + b_2
            interaction_embeddings = torch.matmul(interaction_embeddings, self.weight_dict['W_two_%d' % k]) + self.weight_dict['b_two_%d' % k] #?????? : use torch.matmul, self.weight_dict['W_two_%d' % k], self.weight_dict['b_two_%d' % k]

            # non-linear activation 
            ego_embeddings = simple_embeddings + interaction_embeddings #??????: use simple_embeddings, interaction_embeddings
                        
            
            # add message dropout
            mess_dropout_mask = nn.Dropout(self.mess_dropout)
            ego_embeddings = mess_dropout_mask(ego_embeddings)

            # Perform L2 normalization
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            
            final_embeddings.append(norm_embeddings)                                            

        
        final_embeddings = torch.cat(final_embeddings, 1)                           
    
        # back to user/item dimension
        u_final_embeddings, i_final_embeddings = final_embeddings.split([self.n_users, self.n_items], 0)

        self.u_final_embeddings = nn.Parameter(u_final_embeddings)
        self.i_final_embeddings = nn.Parameter(i_final_embeddings)
        
        u_emb = u_final_embeddings[u] # user embeddings
        p_emb = i_final_embeddings[i] # positive item embeddings
        n_emb = i_final_embeddings[j] # negative item embeddings
        
        y_ui =  torch.sum(torch.mul(u_emb, p_emb), axis=1)               # ?????? : use torch.mul, sum() method                             
        y_uj =  torch.sum(torch.mul(u_emb, n_emb), axis=1)               # ?????? : use torch.mul, sum() method 
        
        
        log_prob =  torch.log(torch.sigmoid(y_ui-y_uj)).mean()           # ?????? : use torch.log, torch.sigmoid, mean() method
        bpr_loss = -log_prob        
        if self.reg > 0.:
            l2norm = (torch.norm(u_emb**2) 
                    + torch.norm(p_emb**2) 
                    + torch.norm(n_emb**2)) / 2 #
            l2reg = self.reg*l2norm / u_emb.shape[0]      
            bpr_loss += l2reg
        
        return bpr_loss
