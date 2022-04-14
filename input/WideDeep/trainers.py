import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim import Adam

from utils import ndcg_k, recall_at_k


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.submission_dataloader = submission_dataloader

        # self.data_name = self.args.data_name
        # betas = (self.args.adam_beta1, self.args.adam_beta2)
        # self.optim = Adam(
        #     self.model.parameters(),
        #     lr=self.args.lr,
        #     betas=betas,
        #     weight_decay=self.args.weight_decay,
        # )
        self.optim = Adam(model.parameters(), lr= self.args.lr)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, mode="valid")

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, mode="test")

    def submission(self, epoch):
        return self.iteration(epoch, self.submission_dataloader, mode="submission")

    def iteration(self, epoch, dataloader, mode="train"):
        raise NotImplementedError

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "RECALL@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "RECALL@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
        }
        print(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    # def cross_entropy(self, seq_out, pos_ids, neg_ids):
    #     # [batch ,  seq_len , hidden_size]
    #     pos_emb = self.model.item_embeddings(pos_ids)
    #     neg_emb = self.model.item_embeddings(neg_ids)
    #     # [batch*seq_len , hidden_size] , 순서는 그대로 유지하되 일렬로 펼침(3차원에서 2차원으로)
    #     pos = pos_emb.view(-1, pos_emb.size(2))
    #     neg = neg_emb.view(-1, neg_emb.size(2))
    #     seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len , hidden_size]
    #     pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len] , 클수록 좋고
    #     neg_logits = torch.sum(neg * seq_emb, -1) # 작을 수록 좋다

    #     istarget = (
    #         (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
    #     )  # 0이 아닌 것들 중에, [batch*seq_len]

    #     loss = torch.sum(
    #         -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
    #         - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
    #     ) / torch.sum(istarget)

    #     return loss

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class PretrainTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):
        super(PretrainTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def pretrain(self, epoch, pretrain_dataloader):

        desc = (
            f"AAP-{self.args.aap_weight}-"
            f"MIP-{self.args.mip_weight}-"
            f"MAP-{self.args.map_weight}-"
            f"SP-{self.args.sp_weight}"
        )

        pretrain_data_iter = tqdm.tqdm(
            enumerate(pretrain_dataloader),
            desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
            total=len(pretrain_dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        self.model.train()
        aap_loss_avg = 0.0
        mip_loss_avg = 0.0
        map_loss_avg = 0.0
        sp_loss_avg = 0.0

        for i, batch in pretrain_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.device) for t in batch)
            (
                attributes,
                masked_item_sequence,
                pos_items,
                neg_items,
                masked_segment_sequence,
                pos_segment,
                neg_segment,
            ) = batch

            aap_loss, mip_loss, map_loss, sp_loss = self.model.pretrain(
                attributes,
                masked_item_sequence,
                pos_items,
                neg_items,
                masked_segment_sequence,
                pos_segment,
                neg_segment,
            )

            joint_loss = (
                self.args.aap_weight * aap_loss
                + self.args.mip_weight * mip_loss
                + self.args.map_weight * map_loss
                + self.args.sp_weight * sp_loss
            )

            self.optim.zero_grad()
            joint_loss.backward()
            self.optim.step()

            aap_loss_avg += aap_loss.item()
            mip_loss_avg += mip_loss.item()
            map_loss_avg += map_loss.item()
            sp_loss_avg += sp_loss.item()

        num = len(pretrain_data_iter) * self.args.pre_batch_size
        losses = {
            "epoch": epoch,
            "aap_loss_avg": aap_loss_avg / num,
            "mip_loss_avg": mip_loss_avg / num,
            "map_loss_avg": map_loss_avg / num,
            "sp_loss_avg": sp_loss_avg / num,
        }
        print(desc)
        print(str(losses))
        return losses


class FinetuneTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )
    # trainer 의 iteration method override
    def iteration(self, epoch, dataloader, mode="train"):

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(
            enumerate(dataloader),
            desc="Recommendation EP_%s:%d" % (mode, epoch),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )
        if mode == "train":
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                x, y = batch
                # Binary cross_entropy
                output = self.model(x)
                loss = self.criterion(output, y.float())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": "{:.4f}".format(rec_cur_loss),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

        else:
            self.model.eval()
            
            correct_result_sum = 0
            count = 0
            precision_sum = 0
            recall_sum = 0
            count_cn = 0
            for i, batch in rec_data_iter:

                batch = tuple(t.to(self.device) for t in batch)
                x,y  = batch
                count += x.shape[0]
                output = self.model(x)
                result = torch.round(output)
                correct_result_sum += (result == y).sum().float()
                # print("outyut.shape :", result.shape)
                # print("output.shape[0] : ", result.shape[0])
                precision , recall = self.confusion(result, y)
                precision_sum += precision
                recall_sum += recall
                count_cn += 1
                # rating_pred = self.predict_full(recommend_output)

                # rating_pred = rating_pred.cpu().data.numpy().copy()
                # batch_user_index = user_ids.cpu().numpy()
                # # implicit 이 1인 경우, rating_pred를 0으로 만들어준다
                # rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                # # arpartition으로 유사도가 높은 Top-10 의 index를 뽑는다
                # ind = np.argpartition(rating_pred, -10)[:, -10:]

                # arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

                # arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

                # batch_pred_list = ind[
                #     np.arange(len(rating_pred))[:, None], arr_ind_argsort
                # ]

                # if i == 0:
                #     pred_list = batch_pred_list
                #     answer_list = answers.cpu().data.numpy()
                # else:
                # #np.append(pred_list 와 batch_pred_list를 행을 기준으로 합친다)
                #     pred_list = np.append(pred_list, batch_pred_list, axis=0)
                #     answer_list = np.append(
                #         answer_list, answers.cpu().data.numpy(), axis=0
                #     )
            acc = (correct_result_sum/count)*100
            precision_mean = (precision_sum/count_cn)
            recall_mean = (recall_sum/count_cn)
            print("Final Acc : {:.2f}%".format(acc.item()))
            print("Final precision : ", precision_mean)
            print("Final recall : ", recall_mean)
            return acc.item(), precision_mean, recall_mean
            # if mode == "submission":
            #     return pred_list
            # else:
            #     return self.get_full_sort_score(epoch, answer_list, pred_list)

    def confusion(self, output, y):
        tp, fp = 0, 0
        fn ,tn = 0, 0
        precision = 0
        recall = 0

        output = output.cpu().data.numpy()
        y = y.cpu().data.numpy()
        for i in range(0,output.shape[0]):
            if y[i,:] == 0:
                if output[i, :] == 0:
                    tn += 1
                else:
                    fp += 1
            elif y[i, :] == 1:
                if output[i, :] == 1:
                    tp += 1
                else:
                    fn += 1
        precision = tp / (tp+fp)
        recall = tp / (tp + fn)
        return precision , recall
        

