import torch
import numpy as np
from torch import nn

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, layers, reg, node_dropout, mess_dropout, adj_mtx, device, mode = 'train'):
        super().__init__()
        self.device = device

        # config, dataset

        # initialize Class attributes TODO (Dataset을 참조하여 불러올 수 있는 부분은 해당 부분을 일괄적으로 수정해주자.)
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.l_matrix = adj_mtx
        # self.l_plus_i_matrix = adj_mtx + sp.eye(adj_mtx.shape[0])
        self.reg = reg
        self.layers = layers
        self.n_layers = len(self.layers)
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout


        # initialize weights
        self.weight_dict = self._init_weights()
        print("Weights initialized.")


        # Create Matrix 'L+I', PyTorch sparse tensor of SP adjacency_mtx
        # self.L_plus_I = self._convert_sp_mat_to_sp_tensor(self.l_plus_i_matrix)
        self.L = self._convert_sp_mat_to_sp_tensor(self.l_matrix)

        # this is for load_state_dict 
        if mode == 'submission':
            self.u_final_embeddings = nn.Parameter(torch.empty(self.n_users, self.emb_dim).to(self.device))
            self.i_final_embeddings = nn.Parameter(torch.empty(self.n_items, self.emb_dim).to(self.device))

    # weights initialization
    def _init_weights(self):
        print("Initializing weights...")
        weight_dict = nn.ParameterDict()
        initializer = torch.nn.init.xavier_uniform_

        weight_dict['user_embedding'] = nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim).to(self.device)))
        weight_dict['item_embedding'] = nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim).to(self.device)))

        # weight_size_list = [self.emb_dim] + self.layers
        # for k in range(self.n_layers):
        #     weight_dict['W_one_%d' %k] = nn.Parameter(initializer(torch.empty(weight_size_list[k], weight_size_list[k+1]).to(self.device)))
        #     weight_dict['b_one_%d' %k] = nn.Parameter(initializer(torch.empty(1, weight_size_list[k+1]).to(self.device)))
            
        #     weight_dict['W_two_%d' %k] = nn.Parameter(initializer(torch.empty(weight_size_list[k], weight_size_list[k+1]).to(self.device)))
        #     weight_dict['b_two_%d' %k] = nn.Parameter(initializer(torch.empty(1, weight_size_list[k+1]).to(self.device)))

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

    
    def forward(self, u, i, j):
        """
        Computes the forward pass
        
        Arguments:
        ---------
        u = user
        i = positive item (user interacted with item)
        j = negative item (user did not interact with item)
        """
        ego_embeddings = torch.cat((self.weight_dict['user_embedding'], self.weight_dict['item_embedding']), 0)
        final_embeddings = [ego_embeddings]

        for layer_idx in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.L, ego_embeddings)
            final_embeddings.append(ego_embeddings)
        lightgcn_all_embeddings = torch.stack(final_embeddings, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        u_final_embeddings, i_final_embeddings = lightgcn_all_embeddings.split([self.n_users, self.n_items], 0)

        self.u_final_embeddings = nn.Parameter(u_final_embeddings)
        self.i_final_embeddings = nn.Parameter(i_final_embeddings)

        u_emb = u_final_embeddings[u] # user embeddings
        p_emb = i_final_embeddings[i] # positive item embeddings
        n_emb = i_final_embeddings[j] # negative item embeddings
        
        y_ui =  torch.sum(torch.mul(u_emb, p_emb), axis=1)                                           
        y_uj =  torch.sum(torch.mul(u_emb, n_emb), axis=1)              
        
        log_prob =  torch.log(torch.sigmoid(y_ui-y_uj)).mean()           
        bpr_loss = -log_prob        
        
        if self.reg > 0.:
            l2norm = (torch.norm(u_emb**2) 
                    + torch.norm(p_emb**2) 
                    + torch.norm(n_emb**2)) / 2 #
            l2reg = self.reg*l2norm / u_emb.shape[0]      
            bpr_loss += l2reg
        
        return bpr_loss
        

