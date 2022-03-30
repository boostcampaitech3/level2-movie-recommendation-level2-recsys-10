# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

""" This model implements the paper "Zhou et al., Deep Interest Network for 
    Click-Through Rate Prediction, KDD'2018".
    [PDF] https://arxiv.org/pdf/1706.06978.pdf
    [Code] https://github.com/zhougr1993/DeepInterestNetwork
"""

import torch
from torch import nn
import numpy as np
import os, sys
import logging
from metrics import evaluate_metrics
from utils import get_device, get_optimizer, get_loss_fn, get_regularizer, Monitor
from layers import EmbeddingDictLayer, MLP_Layer, Dice

class BaseModel(nn.Module):
    def __init__(self, 
                 feature_map, 
                 model_id="BaseModel", 
                 gpu=-1, 
                 monitor="AUC", 
                 save_best_only=True, 
                 monitor_mode="max", 
                 patience=2, 
                 every_x_epochs=1, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 reduce_lr_on_plateau=True, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 **kwargs):
        super(BaseModel, self).__init__()
        self.device = get_device(gpu)
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._patience = patience
        self._every_x_epochs = every_x_epochs # float acceptable
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._embedding_initializer = embedding_initializer
        self._feature_map = feature_map
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model"))
        self._validation_metrics = kwargs["metrics"]
        self._verbose = kwargs["verbose"]

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = get_loss_fn(loss)

    def add_loss(self, inputs, reduction="mean"):
        return_dict = self.forward(inputs)
        loss = self.loss_fn(return_dict["y_pred"], return_dict["y_true"], reduction=reduction)
        return loss

    def add_regularization(self):
        reg_loss = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if "embedding_layer" in name:
                        if self._embedding_regularizer:
                            for emb_p, emb_lambda in emb_reg:
                                reg_loss += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                    else:
                        if self._net_regularizer:
                            for net_p, net_lambda in net_reg:
                                reg_loss += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_loss


    def get_total_loss(self, inputs):
        total_loss = self.add_loss(inputs) + self.add_regularization()
        return total_loss

    def reset_parameters(self):
        def reset_param(m):
            if type(m) == nn.ModuleDict:
                for k, v in m.items():
                    if type(v) == nn.Embedding:
                        if "pretrained_emb" in self._feature_map.feature_specs[k]: # skip pretrained
                            continue
                        if self._embedding_initializer is not None:
                            try:
                                if v.padding_idx is not None:
                                    # the last index is padding_idx
                                    initializer = self._embedding_initializer.replace("(", "(v.weight[0:-1, :],")
                                else:
                                    initializer = self._embedding_initializer.replace("(", "(v.weight,")
                                eval(initializer)
                            except:
                                raise NotImplementedError("embedding_initializer={} is not supported."\
                                                          .format(self._embedding_initializer))
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(reset_param)
        
    def inputs_to_device(self, inputs):
        X, y = inputs
        X = X.to(self.device)
        y = y.float().view(-1, 1).to(self.device)
        self.batch_size = y.size(0)
        return X, y

    def model_to_device(self):
        self.to(device=self.device)

    def on_batch_end(self, batch, logs={}):
        self._total_batches += 1
        if (batch + 1) % self._every_x_batches == 0 or (batch + 1) % self._batches_per_epoch == 0:
            epoch = round(float(self._total_batches) / self._batches_per_epoch, 2)
            val_logs = self.evaluate_generator(self.valid_gen)
            self.checkpoint_and_earlystop(epoch, val_logs)
            logging.info("--- {}/{} batches finished ---".format(batch + 1, self._batches_per_epoch))

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr

    def checkpoint_and_earlystop(self, epoch, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({}) STOP: {:.6f} !".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({}): {:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps * self._every_x_epochs >= self._patience:
            self._stop_training = True
            logging.info("Early stopping at epoch={:g}".format(epoch))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)
            
    def fit_generator(self, data_generator, epochs=1, validation_data=None,
                      verbose=0, max_gradient_norm=10., **kwargs):
        """
        Training a model and valid accuracy.
        Inputs:
        - iter_train: I
        - iter_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        """
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._total_batches = 0
        self._batches_per_epoch = len(data_generator)
        self._every_x_batches = int(np.ceil(self._every_x_epochs * self._batches_per_epoch))
        self._stop_training = False
        self._verbose = verbose
        
        logging.info("Start training: {} batches/epoch".format(self._batches_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(data_generator, epoch)
            logging.info("Train loss: {:.6f}".format(epoch_loss))
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(epoch + 1))
        logging.info("Training finished.")

    def train_one_epoch(self, data_generator, epoch):
        epoch_loss = 0
        self.train()
        batch_iterator = data_generator
        if self._verbose > 0:
            from tqdm import tqdm
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self.optimizer.zero_grad()
            loss = self.get_total_loss(batch_data)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            epoch_loss += loss.item()
            self.on_batch_end(batch_index)
            if self._stop_training:
                break
        return epoch_loss / self._batches_per_epoch

    def evaluate_generator(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            if self._verbose > 0:
                from tqdm import tqdm
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(batch_data[1].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            val_logs = self.evaluate_metrics(y_true, y_pred, self._validation_metrics)
            return val_logs

    def evaluate_metrics(self, y_true, y_pred, metrics):
        return evaluate_metrics(y_true, y_pred, metrics)

    def predict_generator(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            if self._verbose > 0:
                from tqdm import tqdm
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            return y_pred
                
    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)
        del state_dict
        torch.cuda.empty_cache()

    def get_output_activation(self, task="binary_classification"):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "regression":
            return None
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))

class DIN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DIN", 
                 gpu=-1, 
                 task="binary_classification",
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 net_dropout=0, 
                 batch_norm=False, 
                 din_target_field=[("item_id", "cate_id")],
                 din_sequence_field=[("click_history", "cate_history")],
                 din_use_softmax=False,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(DIN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        if not isinstance(din_target_field, list):
            din_target_field = [din_target_field]
        self.din_target_field = din_target_field
        if not isinstance(din_sequence_field, list):
            din_sequence_field = [din_sequence_field]
        self.din_sequence_field = din_sequence_field
        assert len(self.din_target_field) == len(self.din_sequence_field), \
            "Error: len(self.din_target_field) != len(self.din_sequence_field)"
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingDictLayer(feature_map, embedding_dim)
        self.attention_layers = nn.ModuleList(
            [DIN_Attention(embedding_dim * len(target_field) if type(target_field) == tuple \
                                                             else embedding_dim,
                           attention_units=attention_hidden_units,
                           hidden_activations=attention_hidden_activations,
                           output_activation=attention_output_activation,
                           dropout_rate=attention_dropout,
                           batch_norm=batch_norm,
                           use_softmax=din_use_softmax)
             for target_field in self.din_target_field])
        self.dnn = MLP_Layer(input_dim=feature_map.num_fields * embedding_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.get_output_activation(task), 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm, 
                             use_bias=True)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb_dict = self.embedding_layer(X)
        for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field, 
                                                                 self.din_sequence_field)):
            target_emb = self.concat_embedding(target_field, feature_emb_dict)
            sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict)
            seq_field = np.array([sequence_field]).flatten()[0] # pick a sequence field
            padding_idx = self.feature_map.feature_specs[seq_field]['padding_idx']
            mask = (X[:, self.feature_map.feature_specs[seq_field]["index"]].long() != padding_idx).float()
            pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask)
            for field, field_emb in zip(np.hstack([sequence_field]),
                                        pooling_emb.split(self.embedding_dim, dim=-1)):
                feature_emb_dict[field] = field_emb
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
        y_pred = self.dnn(feature_emb.flatten(start_dim=1))
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

    def concat_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]


class DIN_Attention(nn.Module):
    def __init__(self, 
                 embedding_dim=64,
                 attention_units=[32], 
                 hidden_activations="ReLU",
                 output_activation=None,
                 dropout_rate=0,
                 batch_norm=False,
                 use_softmax=False):
        super(DIN_Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_softmax = use_softmax
        if isinstance(hidden_activations, str) and hidden_activations.lower() == "dice":
            hidden_activations = [Dice(units) for units in attention_units]
        self.attention_layer = MLP_Layer(input_dim=4 * embedding_dim,
                                         output_dim=1,
                                         hidden_units=attention_units,
                                         hidden_activations=hidden_activations,
                                         output_activation=output_activation,
                                         dropout_rates=dropout_rate,
                                         batch_norm=batch_norm, 
                                         use_bias=True)

    def forward(self, target_item, history_sequence, mask=None):
        # target_item: b x emd
        # history_sequence: b x len x emb
        seq_len = history_sequence.size(1)
        target_item = target_item.unsqueeze(1).expand(-1, seq_len, -1)
        attention_input = torch.cat([target_item, history_sequence, target_item - history_sequence, 
                                     target_item * history_sequence], dim=-1) # b x len x 4*emb
        attention_weight = self.attention_layer(attention_input.view(-1, 4 * self.embedding_dim))
        attention_weight = attention_weight.view(-1, seq_len) # b x len
        if mask is not None:
            attention_weight = attention_weight * mask.float()
        if self.use_softmax:
            if mask is not None:
                attention_weight += -1.e9 * (1 - mask.float())
            attention_weight = attention_weight.softmax(dim=-1)
        output = (attention_weight.unsqueeze(-1) * history_sequence).sum(dim=1)
        return output