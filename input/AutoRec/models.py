import torch
import torch.nn as nn

from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy

import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (
    BM25Recommender,
    CosineRecommender,
    TFIDFRecommender,
    bm25_weight,
)

from modules import Encoder, LayerNorm, activation_layer, MLPLayers
from utils import mf_sgd, get_predicted_full_matrix, get_rmse, item_encoding, als, get_ALS_loss



class S3RecModel(nn.Module):
    def __init__(self, args):
        super(S3RecModel, self).__init__()
        self.item_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0
        )
        self.attribute_embeddings = nn.Embedding(
            args.attribute_size, args.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.mip_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.map_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.sp_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # AAP
    def associated_attribute_prediction(self, sequence_output, attribute_embedding):
        """
        :param sequence_output: [B L H]
        :param attribute_embedding: [arribute_num H]
        :return: scores [B*L tag_num]
        """
        sequence_output = self.aap_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view(
            [-1, self.args.hidden_size, 1]
        )  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L tag_num]

    # MIP sample neg items
    def masked_item_prediction(self, sequence_output, target_item):
        """
        :param sequence_output: [B L H]
        :param target_item: [B L H]
        :return: scores [B*L]
        """
        sequence_output = self.mip_norm(
            sequence_output.view([-1, self.args.hidden_size])
        )  # [B*L H]
        target_item = target_item.view([-1, self.args.hidden_size])  # [B*L H]
        score = torch.mul(sequence_output, target_item)  # [B*L H]
        return torch.sigmoid(torch.sum(score, -1))  # [B*L]

    # MAP
    def masked_attribute_prediction(self, sequence_output, attribute_embedding):
        sequence_output = self.map_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view(
            [-1, self.args.hidden_size, 1]
        )  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L tag_num]

    # SP sample neg segment
    def segment_prediction(self, context, segment):
        """
        :param context: [B H]
        :param segment: [B H]
        :return:
        """
        context = self.sp_norm(context)
        score = torch.mul(context, segment)  # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1))  # [B]

    #
    def add_position_embedding(self, sequence, args):

        seq_length = sequence.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=sequence.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)

        if not args.rm_position:
            position_embeddings = self.position_embeddings(position_ids)
            sequence_emb = item_embeddings + position_embeddings
        else:
            sequence_emb = item_embeddings

        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def pretrain(
        self,
        attributes,
        masked_item_sequence,
        pos_items,
        neg_items,
        masked_segment_sequence,
        pos_segment,
        neg_segment,
    ):

        # Encode masked sequence

        sequence_emb = self.add_position_embedding(masked_item_sequence)
        sequence_mask = (masked_item_sequence == 0).float() * -1e8
        sequence_mask = torch.unsqueeze(torch.unsqueeze(sequence_mask, 1), 1)

        encoded_layers = self.item_encoder(
            sequence_emb, sequence_mask, output_all_encoded_layers=True
        )
        # [B L H]
        sequence_output = encoded_layers[-1]

        attribute_embeddings = self.attribute_embeddings.weight
        # AAP
        aap_score = self.associated_attribute_prediction(
            sequence_output, attribute_embeddings
        )
        aap_loss = self.criterion(
            aap_score, attributes.view(-1, self.args.attribute_size).float()
        )
        # only compute loss at non-masked position
        aap_mask = (masked_item_sequence != self.args.mask_id).float() * (
            masked_item_sequence != 0
        ).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))

        # MIP
        pos_item_embs = self.item_embeddings(pos_items)
        neg_item_embs = self.item_embeddings(neg_items)
        pos_score = self.masked_item_prediction(sequence_output, pos_item_embs)
        neg_score = self.masked_item_prediction(sequence_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.criterion(
            mip_distance, torch.ones_like(mip_distance, dtype=torch.float32)
        )
        mip_mask = (masked_item_sequence == self.args.mask_id).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        # MAP
        map_score = self.masked_attribute_prediction(
            sequence_output, attribute_embeddings
        )
        map_loss = self.criterion(
            map_score, attributes.view(-1, self.args.attribute_size).float()
        )
        map_mask = (masked_item_sequence == self.args.mask_id).float()
        map_loss = torch.sum(map_loss * map_mask.flatten().unsqueeze(-1))

        # SP
        # segment context
        segment_context = self.add_position_embedding(masked_segment_sequence)
        segment_mask = (masked_segment_sequence == 0).float() * -1e8
        segment_mask = torch.unsqueeze(torch.unsqueeze(segment_mask, 1), 1)
        segment_encoded_layers = self.item_encoder(
            segment_context, segment_mask, output_all_encoded_layers=True
        )

        # take the last position hidden as the context
        segment_context = segment_encoded_layers[-1][:, -1, :]  # [B H]
        # pos_segment
        pos_segment_emb = self.add_position_embedding(pos_segment)
        pos_segment_mask = (pos_segment == 0).float() * -1e8
        pos_segment_mask = torch.unsqueeze(torch.unsqueeze(pos_segment_mask, 1), 1)
        pos_segment_encoded_layers = self.item_encoder(
            pos_segment_emb, pos_segment_mask, output_all_encoded_layers=True
        )
        pos_segment_emb = pos_segment_encoded_layers[-1][:, -1, :]

        # neg_segment
        neg_segment_emb = self.add_position_embedding(neg_segment)
        neg_segment_mask = (neg_segment == 0).float() * -1e8
        neg_segment_mask = torch.unsqueeze(torch.unsqueeze(neg_segment_mask, 1), 1)
        neg_segment_encoded_layers = self.item_encoder(
            neg_segment_emb, neg_segment_mask, output_all_encoded_layers=True
        )
        neg_segment_emb = neg_segment_encoded_layers[-1][:, -1, :]  # [B H]

        pos_segment_score = self.segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self.segment_prediction(segment_context, neg_segment_emb)

        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)

        sp_loss = torch.sum(
            self.criterion(
                sp_distance, torch.ones_like(sp_distance, dtype=torch.float32)
            )
        )

        return aap_loss, mip_loss, map_loss, sp_loss

    # Fine tune
    # same as SASRec
    def finetune(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids, self.args)

        item_encoded_layers = self.item_encoder(
            sequence_emb, extended_attention_mask, output_all_encoded_layers=True
        )

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class BERT4RecModel(nn.Module):
    def __init__(self, args):
        super(BERT4RecModel, self).__init__()
        self.item_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.out = nn.Linear(args.hidden_size, args.item_size - 1)
        self.args = args

        #self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    def add_position_embedding(self, sequence, args):

        seq_length = sequence.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=sequence.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)

        if not args.rm_position:
            position_embeddings = self.position_embeddings(position_ids)
            sequence_emb = item_embeddings + position_embeddings
        else:
            sequence_emb = item_embeddings

        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).repeat(1, input_ids.shape[1], 1).unsqueeze(1).long()

        if self.args.cuda_condition:
            extended_attention_mask = extended_attention_mask.cuda()

        # extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids, self.args)

        item_encoded_layers = self.item_encoder(
            sequence_emb, extended_attention_mask, output_all_encoded_layers=True
        )

        sequence_output = item_encoded_layers[-1]
        sequence_output = self.out(sequence_output)

        return sequence_output

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class MF(object):
    
    def __init__(self, args):
        self.args = args
        self.R = self.args.train_matrix
        self.num_users, self.num_items = self.R.shape
        self.hidden_size = self.args.hidden_size_mf
        self.lr = self.args.lr
        self.l2_reg = self.args.l2_reg_mf
        self.data_file = self.args.data_file
        
        # ??????, ????????? ?????? ?????? ?????? ?????????
        self.P = np.random.normal(scale=1./self.hidden_size, size=(self.num_users, self.hidden_size))
        self.Q = np.random.normal(scale=1./self.hidden_size, size=(self.num_items, self.hidden_size))

        # ?????????, ??????, ????????? bias ?????????
        self.b = 0.0
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)

        # ?????? ????????? ??????
        self.I, self.J, self.V = scipy.sparse.find(self.R)
        self.samples = [
            (i, j, v) for i, j, v in tqdm(zip(self.I, self.J, self.V), total = len(self.I))]


    def train(self):
        
        np.random.shuffle(self.samples)
        mf_sgd(self.P, self.Q, self.b, self.b_u, self.b_i, self.samples, self.lr, self.l2_reg)
        predicted_R = self.get_predicted_full_matrix()
        rmse = get_rmse(self.R, predicted_R)
        
        return rmse
    
    def submission(self):
        df = pd.read_csv(self.data_file)
        rating_df = item_encoding(df, self.args.model_name)
        items = rating_df['item_idx'].unique()
        users = rating_df['user_idx'].unique()

        predicted_user_item_matrix = pd.DataFrame(self.get_predicted_full_matrix(), columns=items, index=users)

        for idx, user in enumerate(tqdm(users)):

            rating_pred = predicted_user_item_matrix.loc[user].values

            rating_pred[self.R[idx].toarray().reshape(-1) > 0] = 0

            ind = np.argpartition(rating_pred, -10)[-10:]

            ind_argsort = np.argsort(rating_pred[ind])[::-1]

            user_pred_list = ind[ind_argsort].reshape(1, -1)
            
            if idx == 0:
                pred_list = user_pred_list

            else:
                pred_list = np.append(pred_list, user_pred_list, axis=0)


        return pred_list

    def get_predicted_full_matrix(self):
        return get_predicted_full_matrix(self.P, self.Q, self.b, self.b_u, self.b_i)

class MF_ALS(object):
    
    def __init__(self, args):
        self.args = args
        self.F = self.args.train_matrix.toarray()
        self.num_users, self.num_items = self.args.train_matrix.shape
        self.hidden_size = self.args.hidden_size_als
        self.alpha = self.args.alpha
        self.C = 1 + self.alpha * np.copy(self.F)
        self.l2_reg = self.args.l2_reg_als
        self.data_file = self.args.data_file
        
        # ??????, ????????? ?????? ?????? ?????? ?????????
        self.P = np.random.normal(scale=1./self.hidden_size, size=(self.num_users, self.hidden_size))
        self.Q = np.random.normal(scale=1./self.hidden_size, size=(self.num_items, self.hidden_size))
    
    def train(self):
        als(self.F, self.P, self.Q, self.C, self.hidden_size, self.l2_reg)
        loss = get_ALS_loss(self.F, self.P, self.Q, self.C, self.l2_reg)
        
        return loss
    
    def submission(self):
        df = pd.read_csv(self.data_file)
        rating_df = item_encoding(df, self.args.model_name)
        items = rating_df['item_idx'].unique()
        users = rating_df['user_idx'].unique()

        predicted_user_item_matrix = pd.DataFrame(self.get_predicted_full_matrix(), columns=items, index=users)

        for idx, user in enumerate(tqdm(users)):

            rating_pred = predicted_user_item_matrix.loc[user].values

            rating_pred[self.R[idx].toarray().reshape(-1) > 0] = 0

            ind = np.argpartition(rating_pred, -10)[-10:]


            ind_argsort = np.argsort(rating_pred[ind])[::-1]

            user_pred_list = ind[ind_argsort].reshape(1, -1)

            
            if idx == 0:
                pred_list = user_pred_list

            else:
                pred_list = np.append(pred_list, user_pred_list, axis=0)

        return pred_list

    def get_predicted_full_matrix(self):
        return get_predicted_full_matrix(self.P, self.Q)

class Implicit_model(object):
    
    def __init__(self, args):
        self.args = args
        self.user_item_data = self.args.train_matrix
        self.num_users, self.num_items = self.user_item_data.shape
        self.factors = self.args.hidden_size
        self.regularization = self.args.regularization
        self.iterations = self.args.iterations
        self.calculate_training_loss = self.args.calculate_loss
        self.random_state = self.args.seed
        self.data_file = self.args.data_file
        self.model_name = self.args.model_name
        self.bm25 = self.args.bm25
        self.B = self.args.bm25_B
        self.lr = self.args.lr
        self.verify_negative_samples = self.args.verify_negative_samples
        self.neg_prop = self.args.neg_prop

        if self.model_name == 'ALS':
            self.model = AlternatingLeastSquares(
                factors = self.factors,
                regularization = self.regularization,
                use_gpu = self.args.cuda_condition,
                iterations = self.iterations,
                calculate_training_loss = self.calculate_training_loss,
                random_state = self.random_state
            )
        elif self.model_name == 'BPR':
            self.model = BayesianPersonalizedRanking(
                factors = self.factors,
                learning_rate = self.lr,
                regularization = self.regularization,
                use_gpu = self.args.cuda_condition,
                iterations = self.iterations,
                verify_negative_samples = self.verify_negative_samples,
                random_state = self.random_state
            )

        elif self.model_name == 'LMF':
            self.model = LogisticMatrixFactorization(
                factors = self.factors,
                learning_rate = self.lr,
                regularization = self.regularization,
                iterations = self.iterations,
                neg_prop = self.neg_prop,
                random_state = self.random_state,
            )
        
        elif self.model_name == 'TFIDF':
            self.model = TFIDFRecommender()
        
        elif self.model_name == 'COSINE':
            self.model = CosineRecommender()

        elif self.model_name == 'BM25':
            self.model = BM25Recommender(B=self.B)


        if self.bm25:
            print("weighting matrix by bm25_weight")
            self.user_item_data = (bm25_weight(self.user_item_data, B=self.B)).tocsr()

    def train(self):
        self.model.fit(self.user_item_data)
    
    def submission(self):
        df = pd.read_csv(self.data_file)
        rating_df = item_encoding(df, self.args.model_name)
        items = rating_df['item_idx'].unique()
        users = rating_df['user_idx'].unique()

        for idx, user in enumerate(tqdm(users)):

            pred_items, pred_scores = self.model.recommend(user, self.user_item_data[user])
            # pred_items, pred_scores = self.model.recommend(user, self.user_item_data[user], N=20)
            
            if idx == 0:
                pred_list = [pred_items]

            else:
                pred_list = np.append(pred_list, [pred_items], axis=0)

        return pred_list

class AutoRec(nn.Module):
    """
    AutoRec
    
    Args:
        - input_dim: (int) input feature??? Dimension
        - emb_dim: (int) Embedding??? Dimension
        - hidden_activation: (str) hidden layer??? activation function.
        - out_activation: (str) output layer??? activation function.
    Shape:
        - Input: (torch.Tensor) input features,. Shape: (batch size, input_dim)
        - Output: (torch.Tensor) reconstructed features. Shape: (batch size, input_dim)
    """
    def __init__(self, args):
        super(AutoRec, self).__init__()
        
        # initialize Class attributes
        self.args = args
        self.input_dim = self.args.input_dim
        self.emb_dim = self.args.hidden_size
        self.hidden_activation = self.args.hidden_activation
        self.out_activation = self.args.out_activation
        self.num_layers = self.args.num_layers
        self.dropout_rate = self.args.dropout_rate
        
        # define layers
        encoder_modules = list()
        encoder_layers = [self.input_dim] + [self.emb_dim // (2 ** i) for i in range(self.num_layers)]
        for i in range(self.num_layers):
            input_size = encoder_layers[i] 
            output_size = encoder_layers[i + 1] 
            encoder_modules.append(nn.Linear(input_size, output_size))
            activation_function = activation_layer(self.hidden_activation)
            if activation_function is not None:
                encoder_modules.append(activation_function)
        encoder_modules.append(nn.Dropout(self.dropout_rate))
        
        decoder_modules = list()
        decoder_layers = encoder_layers[::-1]
        for i in range(self.num_layers):
            input_size = decoder_layers[i] 
            output_size = decoder_layers[i + 1] 
            decoder_modules.append(nn.Linear(input_size, output_size))
            activation_function = activation_layer(self.out_activation)
            if activation_function is not None and (i < self.num_layers - 1):
                decoder_modules.append(activation_function)

        self.encoder = nn.Sequential(
            *encoder_modules
        )
        self.decoder = nn.Sequential(
            *decoder_modules
        )
        
        self.init_weights()

    # initialize weights
    def init_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight.data)
                layer.bias.data.zero_()
        
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight.data)
                layer.bias.data.zero_()

    
    def forward(self, input_feature):
        h = self.encoder(input_feature)
        output = self.decoder(h)
        
        return output

class NCF(nn.Module):

    def __init__(self, args):
        super(NCF, self).__init__()
        
        # initialize Class attributes
        self.args = args
        self.user_item_data = self.args.train_matrix
        self.n_users, self.n_items = self.user_item_data.shape
        self.emb_dim = self.args.hidden_size
        self.num_layers = self.args.num_layers
        self.layers = [self.emb_dim * 2 // (2 ** i) for i in range(self.num_layers + 1)]
        self.dropout = self.args.dropout_rate
        self.activation = self.args.hidden_activation
        
        # define layers
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim) 
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim) 
        self.mlp_layers = MLPLayers(self.layers, self.dropout, self.activation)
        self.predict_layer = nn.Linear(self.layers[-1] , 1) 
        self.activation_layer = activation_layer(self.activation)
        
        self._init_weights()
        
    # initialize weights
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1, nonlinearity="sigmoid")
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    
    def forward(self, user, item):   
        user_emb = self.user_embedding(user) 
        item_emb = self.item_embedding(item) 
        
        input_feature = torch.cat((user_emb, item_emb), -1) 
        mlp_output =  self.mlp_layers(input_feature) 
        output = self.predict_layer(mlp_output) 
        output = self.activation_layer(output)
        return output.squeeze(-1)

class GMF(nn.Module):
    def __init__(self, args):
        super(GMF, self).__init__()
        self.args = args
        self.user_item_data = self.args.train_matrix
        self.n_users, self.n_items = self.user_item_data.shape
        self.emb_dim = self.args.hidden_size
        self.activation = self.args.hidden_activation

        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        
        self.predict_layer = nn.Linear(self.emb_dim, 1, bias = False)
        self.activation_layer = activation_layer(self.activation)
        
        self._init_weights()
        
    # initialize weights
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1, nonlinearity="sigmoid")
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        gmf = user_emb * item_emb

        output = self.predict_layer(gmf) 
        output = self.activation_layer(output)

        return output.squeeze(-1)

class NeuMF(nn.Module):

    def __init__(self, args, NCF, GMF):
        super(NeuMF, self).__init__()
        
        # initialize Class attributes
        self.args = args
        self.NCF = NCF
        self.GMF = GMF

        self.activation = self.args.hidden_activation
        
        # predict_layer input 
        self.NCF_predict_input = self.NCF.predict_layer.weight.shape[1]
        self.GMF_predict_input = self.GMF.predict_layer.weight.shape[1]
        self.predict_input = self.NCF_predict_input + self.GMF_predict_input

        # define layers
        self.GMF_user_embedding = self.GMF.user_embedding
        self.GMF_item_embedding = self.GMF.item_embedding
        self.NCF_user_embedding = self.NCF.user_embedding
        self.NCF_item_embedding = self.NCF.item_embedding  
        self.mlp_layers = self.NCF.mlp_layers
        self.predict_layer = nn.Linear(self.predict_input, 1, bias = False) 
        self.activation_layer = activation_layer(self.activation)
        
        self._init_weights()
        
    # initialize weights
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1, nonlinearity="sigmoid")
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, user, item):   
        GMF_user_emb = self.GMF_user_embedding(user) 
        GMF_item_emb = self.GMF_item_embedding(item)
        gmf = GMF_user_emb * GMF_item_emb

        NCF_user_emb = self.NCF_user_embedding(user) 
        NCF_item_emb = self.NCF_item_embedding(item)
        
        NCF_input_feature = torch.cat((NCF_user_emb, NCF_item_emb), -1) 
        ncf_output = self.mlp_layers(NCF_input_feature)
        gmf_ncf_concat = torch.cat((gmf, ncf_output), -1) 

        output = self.predict_layer(gmf_ncf_concat) 
        output = self.activation_layer(output)
        return output.squeeze(-1)