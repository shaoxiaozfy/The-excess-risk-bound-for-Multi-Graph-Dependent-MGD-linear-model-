import numpy as np
from torch import nn
import torch
from torch.nn import BCEWithLogitsLoss
from itertools import product


class SurrogateLossForRankingLoss(nn.Module):
    def __init__(self, mode="u1"):
        super().__init__()
        self.mode = mode
        # self.bce = BCEWithLogitsLoss()

    def forward(self, pred_y, true_y):
        # pred_y: tensor, shape: (N,C) N is the batch size and C is the class numbers
        # true_y: tensor, shape: (N,C) N is the batch size and C is the class numbers
        device = true_y.device
        K = true_y.shape[-1]
        n = true_y.shape[0]
        # S_pos = torch.where(true_y == 1)
        # S_neg = torch.where(true_y == 0)

        # (reweighted) univariate loss
        if self.mode == "u1" or self.mode == "u2" or self.mode == "u3" or self.mode == "u4":
            # l = self.bce(pred_y,true_y)
            C = self.calcculate_cost_matrix(true_y).to(device)
            true_y[true_y < 1] = -1

            # print(f"base loss : {self.base_loss(true_y * pred_y)}, C: {C}")

            tmp_loss = self.base_loss(true_y * pred_y) * C
            l = torch.sum(tmp_loss) / K


        # pairwise loss
        elif self.mode=="pa":
            losses = []
            for i in range(K):
                S_pos = torch.where(true_y[:,i] == 1)
                S_neg = torch.where(true_y[:,i] < 1)
                num_pos = len(S_pos[0])
                num_neg = len(S_neg[0]) 
                if num_neg == 0 or num_pos == 0:
                    continue

                mask_pos = true_y[:,i] == 1
                mask_neg = true_y[:,i] < 1
                p = torch.masked_select(pred_y[:,i], mask=mask_pos)
                q = torch.masked_select(pred_y[:,i], mask=mask_neg)
                prod = torch.cartesian_prod(p,q)
                losses.append(torch.sum(self.base_loss(prod[:,0]-prod[:,1])) / (num_pos * num_neg))
            l = torch.sum(torch.stack(losses)) / K

            #     S_pos = torch.where(true_y[i,:] == 1)
            #     S_neg = torch.where(true_y[i,:] < 1)
            #     num_pos = len(S_pos[0])
            #     num_neg = len(S_neg[0]) 
            #     if num_neg == 0 or num_pos == 0:
            #         continue

            #     mask_pos = true_y[i,:] == 1
            #     mask_neg = true_y[i,:] < 1
            #     p = torch.masked_select(pred_y[i,:], mask=mask_pos)
            #     q = torch.masked_select(pred_y[i,:], mask=mask_neg)
            #     prod = torch.cartesian_prod(p,q)
            #     losses.append(torch.sum(self.base_loss(prod[:,0]-prod[:,1])) / (num_pos * num_neg))
            # l = torch.sum(torch.stack(losses)) / n
        else:
            raise NotImplementedError

        return l

    def base_loss(self, ele):
        return torch.log(torch.add(1, torch.exp(-ele)))

    def calcculate_cost_matrix(self, true_y):
        device = true_y.device
        num_instance = true_y.shape[0]
        num_label = true_y.shape[1]
        C = torch.ones(num_instance, num_label).to(device)
        true_y[true_y < 1] = 0

        if self.mode == "u1":
            C = C / num_instance
        elif self.mode == "u2":
            for i in range(num_instance):
                tmp_positive = torch.sum(true_y[i,:])
                tmp_negative = num_label - tmp_positive
                if tmp_positive == 0 or tmp_negative == 0:
                    C[i,:] = 0
                else:
                    C[i,:] = C[i,:]/ (tmp_negative*tmp_positive)
            # tmp_positive = true_y.sum(dim=1)
            # tmp_negative = num_label - tmp_positive

            # mask_nonzero = (tmp_positive != 0) & (tmp_negative != 0)
            # denominator = tmp_positive * tmp_negative

            # C[mask_nonzero, :] = C[mask_nonzero, :] / denominator[mask_nonzero].view(-1, 1)
            # C[~mask_nonzero, :] = 0
        elif self.mode == "u3":
            tmp_positive = true_y.sum(dim=1)
            tmp_negative = num_label - tmp_positive

            mask_nonzero = (tmp_positive != 0) & (tmp_negative != 0)

            for i in range(num_instance):
                if mask_nonzero[i]:
                    pos_mask = true_y[i] == 1
                    C[i, pos_mask] = 1 / tmp_positive[i]

                    neg_mask = true_y[i] == 0
                    C[i, neg_mask] = 1 / tmp_negative[i]
            C[~mask_nonzero, :] = 0

        elif self.mode == "u4":
            tmp_positive = true_y.sum(dim=1)
            tmp_negative = num_label - tmp_positive

            mask_nonzero = (tmp_positive != 0) & (tmp_negative != 0)

            denominator = torch.min(tmp_positive, tmp_negative)

            C[mask_nonzero, :] = C[mask_nonzero, :] / denominator[mask_nonzero].view(-1, 1)
            C[~mask_nonzero, :] = 0
        else:
            raise NotImplementedError
        return C


    def calcculate_cost_matrix_old(self, true_y):
        device = true_y.device
        num_instance = true_y.shape[0]
        num_label = true_y.shape[1]
        C = torch.ones(num_instance, num_label).to(device)
        true_y[true_y < 1] = 0

        if self.mode == "u1":
            for i in range(num_instance):
                C[i,:] = C[i,:] / num_label
        elif self.mode == "u2":
            for i in range(num_instance):
                tmp_postive = torch.sum(true_y[i,:])
                tmp_negative = num_label - tmp_postive
                if tmp_postive == 0 or tmp_negative == 0:
                    C[i,:] = 0
                else:
                    C[i,:] = C[i,:] / (tmp_postive * tmp_negative)
        elif self.mode == "u3":
            for i in range(num_instance):
                tmp_postive = torch.sum(true_y[i,:])
                tmp_negative = num_label - tmp_postive
                if tmp_postive == 0 or tmp_negative == 0:
                    C[i,:] = 0
                else:
                    for j in range(num_label):
                        if true_y[i,j] == 1:
                            C[i,j] = 1 / tmp_postive
                        else:
                            C[i,j] = 1 / tmp_negative
        elif self.mode == "u4":
            for i in range(num_instance):
                tmp_postive = torch.sum(true_y[i,:])
                tmp_negative = num_label - tmp_postive
                if tmp_postive == 0 or tmp_negative == 0:
                    C[i,:] = 0
                else:
                    C[i,:] = C[i,:] / torch.min(tmp_postive, tmp_negative)
        else:
            raise NotImplementedError
        return C
