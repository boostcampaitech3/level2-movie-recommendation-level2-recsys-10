# model - 
# ref. https://github.com/ilya-shenbin/RecVAE/blob/master/model.
# paper: https://arxiv.org/abs/1912.11160

# initialization - 
# ref. https://arxiv.org/abs/1805.08266


import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from zmq import device


def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
        super(CompositePrior, self).__init__()
        
        self.mixture_weights = mixture_weights
        
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)
        
    def forward(self, x, z):
        x = F.normalize(x)
        # h = F.dropout(t, 0, training=self.training)

        post_mu, post_logvar = self.encoder_old(x) # self.encoder_old(x, 0)
        
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        
        density_per_gaussian = torch.stack(gaussians, dim=-1)
                
        return torch.logsumexp(density_per_gaussian, dim=-1)

    
class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=0.1):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                # nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # norm = x.pow(2).sum(dim=-1).sqrt()
        # x = x / norm[:, None]

        # x = F.dropout(x, p=dropout_rate, training=self.training)        

        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)
    

class RecVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, dropout_rate):
        super(RecVAE, self).__init__()

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        
        self.dropout_rate = dropout_rate

        # initialize parameters
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.constant_(self.decoder.bias, 0)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, input):

        mu, logvar = self.encoder(input, dropout_rate=self.dropout_rate)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z) # recon_x

        return x_pred, mu, logvar

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))

    def loss_function(self, recon_x, x, mu, logvar, beta=None, gamma=1):
        """_summary_
        Loss function for RecVAE

        Args:
            recon_x (_type_): output of model (same as x_pred)
            x (_type_): input data
            mu (_type_): _description_
            logvar (_type_): _description_
            gamma (int, optional): _description_. Defaults to 1.
            beta (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if gamma:
            norm = x.sum(dim=-1)
            kl_weight = gamma * norm

        elif beta:
            kl_weight = beta

        z = self.reparameterize(mu, logvar)
        MLL = (F.log_softmax(recon_x, dim=-1) * x).sum(dim=-1).mean()
        KLD = (log_norm_pdf(z, mu, logvar) - self.prior(x, z)).sum(dim=-1).mul(kl_weight).mean()
        negative_elbo = -(MLL - KLD)

        return (MLL, KLD), negative_elbo


class RaCTRecVAE(RecVAE):
    def __init__(self, hidden_dim, latent_dim, input_dim, dropout_rate, pre_model_path, train_stage):
        self.train_stage = train_stage # one of 'actor_pretrain', 'critic_pretrain', 'finetune'
        self.pre_model_path = pre_model_path# TODO 프리트레인된 모델 저장
        super().__init__(hidden_dim, latent_dim, input_dim, dropout_rate)

        # self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        # self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        # self.decoder = nn.Linear(latent_dim, input_dim)
        
        # self.dropout_rate = dropout_rate

        self.number_of_seen_items = 0
        self.number_of_unseen_items = 0

        self.critic_layers = [100,100,10]
        self.critic_layer_dims = [3] + self.critic_layers + [1]

        self.input_matrix = None
        self.predict_matrix = None
        self.true_matrix = None
        self.critic_net = self.construct_critic_layers(self.critic_layer_dims)


        # parameters initialization
        assert self.train_stage in ['actor_pretrain', 'critic_pretrain', 'finetune']
        if self.train_stage == 'actor_pretrain':
            nn.init.kaiming_uniform_(self.decoder.weight)
            nn.init.constant_(self.decoder.bias, 0)
            print(f'== train stage: (1) actor pretrain ====================')
            for p in self.critic_net.parameters():
                p.requires_grad = False
        
        elif self.train_stage == 'critic_pretrain':
            # load pretrained model for finetune
            pretrained = torch.load(self.pre_model_path)
            # self.logger.info('Load pretrained model from', self.pre_model_path)
            print(f'== train stage: (2) critic pretrain | load state dict from {self.pre_model_path} ====================')
            self.load_state_dict(pretrained)
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.prior.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False
        else:
            # load pretrained model for finetune
            pretrained = torch.load(self.pre_model_path)
            # self.logger.info('Load pretrained model from', self.pre_model_path)
            print(f'== train stage: (3) finetune | load state dict from {self.pre_model_path} ====================')
            self.load_state_dict(pretrained)
            for p in self.critic_net.parameters():
                p.requires_grad = False


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, input):
        t = F.normalize(input)

        h = F.dropout(t, self.dropout_rate, training=self.training) * (1 - self.dropout_rate)
        self.input_matrix = h
        self.number_of_seen_items = (h != 0).sum(dim=1)  # network input

        mask = (h > 0) * (t > 0)
        self.true_matrix = t * ~mask
        self.number_of_unseen_items = (self.true_matrix != 0).sum(dim=1)  # remaining input

        mu, logvar = self.encoder(h)
        z = self.reparameterize(mu, logvar)

        x_pred = self.decoder(z) # recon_x
        self.predict_matrix = x_pred

        return x_pred, mu, logvar

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))

    def calculate_actor_loss(self, recon_x, x, mu, logvar, beta, gamma):
        """_summary_
        Loss function for RecVAE

        Args:
            recon_x (_type_): output of model (same as x_pred)
            x (_type_): input data
            mu (_type_): _description_
            logvar (_type_): _description_
            gamma (int, optional): _description_. Defaults to 1.
            beta (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if gamma:
            norm = x.sum(dim=-1)
            kl_weight = gamma * norm

        elif beta:
            kl_weight = beta

        z = self.reparameterize(mu, logvar)
        MLL = (F.log_softmax(recon_x, dim=-1) * x).sum(dim=-1)#.mean()
        KLD = (log_norm_pdf(z, mu, logvar) - self.prior(x, z)).sum(dim=-1).mul(kl_weight)#.mean()
        negative_elbo = -(MLL - KLD)

        return negative_elbo

    def calculate_ndcg(self, predict_matrix, true_matrix, input_matrix, k):
        users_num = predict_matrix.shape[0]
        predict_matrix[input_matrix.nonzero(as_tuple=True)] = -np.inf
        _, idx_sorted = torch.sort(predict_matrix, dim=1, descending=True)

        topk_result = true_matrix[np.arange(users_num)[:, np.newaxis], idx_sorted[:, :k]]

        number_non_zero = ((true_matrix > 0) * 1).sum(dim=1)

        tp = 1. / torch.log2(torch.arange(2, k + 2).type(torch.FloatTensor)).to(topk_result.device)
        DCG = (topk_result * tp).sum(dim=1)
        IDCG = torch.Tensor([(tp[:min(n, k)]).sum() for n in number_non_zero]).to(topk_result.device)
        IDCG = torch.maximum(0.1 * torch.ones_like(IDCG).to(IDCG.device), IDCG)

        return DCG / IDCG

    def construct_critic_layers(self, layer_dims):
        mlp_modules = []
        mlp_modules.append(nn.BatchNorm1d(3))
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.ReLU())
            else:
                mlp_modules.append(nn.Sigmoid())
        return nn.Sequential(*mlp_modules)

    def construct_critic_input(self, actor_loss):
        critic_inputs = []
        critic_inputs.append(self.number_of_seen_items)
        critic_inputs.append(self.number_of_unseen_items)
        critic_inputs.append(actor_loss)
        return torch.stack(critic_inputs, dim=1)

    def critic_forward(self, actor_loss):
        h = self.construct_critic_input(actor_loss)
        y = self.critic_net(h)
        y = torch.squeeze(y)
        return y

    def calculate_critic_loss(self, recon_x, x, mu, logvar, beta, gamma):
        actor_loss = self.calculate_actor_loss(recon_x, x, mu, logvar, beta, gamma)
        y = self.critic_forward(actor_loss)
        score = self.calculate_ndcg(self.predict_matrix, self.true_matrix, self.input_matrix, k=100)

        mse_loss = (y - score) ** 2
        return mse_loss

    def calculate_ac_loss(self, recon_x, x, mu, logvar, beta, gamma):
        actor_loss = self.calculate_actor_loss(recon_x, x, mu, logvar, beta, gamma)
        y = self.critic_forward(actor_loss)
        return -1 * y

    def loss_function(self, recon_x, x, mu, logvar, beta, gamma):
        # actor_pretrain
        if self.train_stage == 'actor_pretrain':
            return self.calculate_actor_loss(recon_x, x, mu, logvar, beta, gamma).mean()
        # critic_pretrain
        elif self.train_stage == 'critic_pretrain':
            return self.calculate_critic_loss(recon_x, x, mu, logvar, beta, gamma).mean()
        # finetune
        else:
            return self.calculate_ac_loss(recon_x, x, mu, logvar, beta, gamma).mean()
