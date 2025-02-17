import os
import time
import random
import torch
from torch import autograd
from tqdm import tqdm
from collections import defaultdict

import numpy as np

import criterion
import tool
import mm_loader
import evaluation


class MILK_session(object):

    def __init__(self, env, model, loader):
        self.env = env
        self.model = model
        self.dataset = loader
        self.optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.model.parameters()), 'lr': self.env.args.lr}])
        self.bpr = criterion.BPR()
        self.align = criterion.MSE()
        self.early_stop = 0
        self.best_epoch = 0
        self.total_epoch = 0
        self.best_ndcg = defaultdict(float)
        self.best_hr = defaultdict(float)
        self.best_recall = defaultdict(float)
        self.test_ndcg = defaultdict(float)
        self.test_hr = defaultdict(float)
        self.test_recall = defaultdict(float)
        
    def train_epoch(self):
        t = time.time()
        self.model.train()
        self.total_epoch += 1
        S = mm_loader.PairSample(self.dataset)
        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2]).long()
        users = users.to(self.env.device)
        posItems = posItems.to(self.env.device)
        negItems = negItems.to(self.env.device)
        users, posItems, negItems = tool.shuffle(users, posItems, negItems)
        total_batch = len(users) // self.env.args.batch_size + 1
        all_loss, all_bpr_loss, all_reg_loss = 0., 0., 0.
        all_rbpr_loss, all_align_loss, all_grad_loss = 0., 0., 0.

        for user, pos_item, neg_item, in mm_loader.minibatch(users,
                                                             posItems,
                                                             negItems,
                                                             batch_size=self.env.args.batch_size):


            image_emb, text_emb = self.model.get_multimedia_emb()
            temp_item = torch.unique(pos_item)
            unique_item = []
            if self.env.args.exp_mode == 'mm':
                for i in temp_item:
                    if i.item() not in self.dataset.warm_missing_item_index:
                        unique_item.append(i.item())
            else:
                unique_item = temp_item
            align_loss = self.align(image_emb[unique_item], text_emb[unique_item])

            mix_ration = [[0.5,0.5]]
            for i in range(1):
                lam_1, lam_2 = np.random.dirichlet([self.env.args.alpha, self.env.args.alpha])
                mix_ration.append([lam_1, lam_2])
                mix_ration.append([lam_2, lam_1])
            env_penalty = []
            env_reg = []
            for i in range(len(mix_ration)):
                env_user_emb, env_item_emb = self.model.get_env_emb(mix_ration, i)
                env_bpr_loss, env_reg_loss = self.bpr(env_user_emb, env_item_emb, user, pos_item,
                                            neg_item)                
                env_penalty.append(env_bpr_loss)
                env_reg.append(env_reg_loss)
            penalty_loss = torch.stack(env_penalty).var()
            rbpr_loss = torch.stack(env_penalty).mean()
            # bpr_loss, reg_loss = self.bpr(user_emb, item_emb, user, pos_item, neg_item)
            penalty_loss = self.env.args.penalty_coeff * penalty_loss

            align_loss = self.env.args.align_coeff * align_loss
            reg_loss = self.env.args.reg_coeff * env_reg[0]
            rbpr_loss = 0
            # loss = bpr_loss  + reg_loss  + penalty_loss + rbpr_loss + align_loss
            bpr_loss = env_penalty[0] + env_penalty[1] + env_penalty[2]
            loss = bpr_loss + reg_loss + align_loss + penalty_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            all_loss += loss
            all_bpr_loss += env_penalty[0]
            all_reg_loss += reg_loss
            all_grad_loss += penalty_loss
            all_rbpr_loss += rbpr_loss
            all_align_loss += align_loss
        return all_loss / total_batch, all_bpr_loss / total_batch, all_reg_loss / total_batch, all_grad_loss/total_batch, all_rbpr_loss/total_batch, all_align_loss/total_batch, time.time() - t
 
    def train(self, epochs):
        for epoch in range(self.env.args.ckpt_start_epoch, epochs):
            loss, bpr_loss, reg_loss, grad_loss, rbpr_loss, align_loss, train_time = self.train_epoch()
            print('-' * 30)
            print(
                f'TRAIN:epoch = {epoch}/{epochs} loss_s1 = {loss:.5f}, bpr_loss_s1 = {bpr_loss:.5f}, rbpr_loss = {rbpr_loss:.5f}, grad_loss = {grad_loss:.5f}, align_loss = {align_loss:.5f}, train_time = {train_time:.2f}')


            if epoch % self.env.args.eva_interval == 0:
                self.early_stop += self.env.args.eva_interval
                hr, recall, ndcg, val_time = self.test(mode='val', top_list=eval(self.env.args.topk))

                if self.env.args.tensorboard:
                    for key in hr.keys():
                        self.env.w.add_scalar(
                            f'Val/hr@{key}', hr[key], self.total_epoch)
                        self.env.w.add_scalar(
                            f'Val/recall@{key}', hr[key], self.total_epoch)
                        self.env.w.add_scalar(
                            f'Val/ndcg@{key}', ndcg[key], self.total_epoch)
                key = list(hr.keys())[0]
                print(
                    f'epoch = {epoch} hr@{key} = {hr[key]:.5f}, recall@{key} = {recall[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, val_time = {val_time:.2f}')

                if ndcg[list(hr.keys())[0]] > self.best_ndcg[list(hr.keys())[0]]:
                    thr, trecall, tndcg, test_time = self.test(mode='test', top_list=eval(self.env.args.topk))
                    self.early_stop = 0
                    for key in thr.keys():
                        tool.cprint(
                            f'epoch = {epoch} hr@{key} = {thr[key]:.5f}, recall@{key} = {trecall[key]:.5f}, ndcg@{key} = {tndcg[key]:.5f}, test_time = {test_time:.2f}')
                    tool.cprint('----------------------')

                    for key in hr.keys():
                        self.best_hr[key] = hr[key]
                        self.best_recall[key] = recall[key]
                        self.best_ndcg[key] = ndcg[key]
                    for key in thr.keys():
                        self.test_hr[key] = thr[key]
                        self.test_recall[key] = trecall[key]
                        self.test_ndcg[key] = tndcg[key]
                    if self.env.args.save:
                        self.save_model(epoch)
                        print('save ckpt')
                    self.best_epoch = epoch
                    if self.env.args.log:
                        self.env.val_logger.info(f'EPOCH[{epoch}/{epochs}]')
                        for key in hr.keys():
                            self.env.val_logger.info(
                                f'hr@{key} = {hr[key]:.5f}, recall@{key} = {recall[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, val_time = {val_time:.2f}')

            if self.env.args.log:
                self.env.train_logger.info(
                    f'EPOCH[{epoch}/{epochs}], loss = {loss:.5f}, bpr_loss = {bpr_loss:.5f}, reg_loss = {reg_loss:.5f}')

            if self.env.args.tensorboard:
                self.env.w.add_scalar(f'Train/loss', loss, self.total_epoch)
                self.env.w.add_scalar(
                    f'Train/bpr_loss', bpr_loss, self.total_epoch)
                self.env.w.add_scalar(
                    f'Train/reg_loss', reg_loss, self.total_epoch)

            if self.early_stop > self.env.args.early_stop // 1:
                break


    def test(self, mode='val', top_list=[50]):
        self.model.eval()
        t = time.time()
        user_emb = self.model.user_emb.weight

        image_feat = self.model.image_feat
        text_feat = self.model.text_feat
        item_emb = (self.model.image_linear(image_feat) + self.model.text_linear(text_feat))/2

        user_emb = user_emb.cpu().detach().numpy()
        item_emb = item_emb.cpu().detach().numpy()
        if mode == 'val':
            hr, recall, ndcg = evaluation.num_faiss_evaluate(self.dataset.val_data,
                                                        list(
                                                            self.dataset.val_data.keys()),
                                                        list(
                                                            self.dataset.cold_item_index),
                                                        self.dataset.train_data,
                                                        top_list, user_emb, item_emb)
        else:
            hr, recall, ndcg = evaluation.num_faiss_evaluate(self.dataset.test_data,
                                                             list(
                                                                     self.dataset.test_data.keys()),
                                                            list(
                                                                self.dataset.cold_item_index),
                                                             self.dataset.train_data,
                                                             top_list, user_emb, item_emb)

        return hr, recall, ndcg, time.time() - t

    def save_ckpt(self, path):
        torch.save(self.model.state_dict(), path)

    def save_model(self, current_epoch):
        model_state_file = os.path.join(
            self.env.CKPT_PATH, f'{self.env.args.suffix}_{self.env.args.penalty_coeff}_epoch{current_epoch}.pth')
        self.save_ckpt(model_state_file)
        if self.best_epoch is not None and current_epoch != self.best_epoch:
            old_model_state_file = os.path.join(
                self.env.CKPT_PATH, f'{self.env.args.suffix}_{self.env.args.penalty_coeff}_epoch{current_epoch}.pth')
            if os.path.exists(old_model_state_file):
                os.system('rm {}'.format(old_model_state_file))
