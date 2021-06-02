import os
import time
import torch
import numpy as np
import torch.nn as nn

from models.model import Net, BiLSMT
from transformers import BertTokenizer
from models.get_optim import get_Adam_optim
from .utils import load_pretrained_datasets, multi_acc, load_specific_datasets

ALL_MODELS = {
    'embedding': Net,
    'lstm': BiLSMT
}

class Trainer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>"})
        self.config.vocab_size = self.tokenizer.vocab_size + 2
        if config.run_mode == 'pretrain':
            self.train_itr, self.dev_itr, self.test_itr = load_pretrained_datasets(self.config, self.tokenizer)
            net = ALL_MODELS["embedding"](config)
            p_net = ALL_MODELS["embedding"](config)
            s_net = None
        else:
            self.train_itr, self.dev_itr, self.test_itr = load_specific_datasets(self.config, self.tokenizer)
            self.config.num_sentes = 4
            p_net = ALL_MODELS["embedding"](config)
            s_net = ALL_MODELS[self.config.model](config)
            net = nn.ModuleDict({'p_net':p_net, 's_net': s_net}) if config.fine_tune == True else s_net

        self.optim = get_Adam_optim(config, net)

        if self.config.n_gpu > 1:
            self.net = torch.nn.DataParallel(net).to(self.config.device)
            self.p_net = torch.nn.DataParallel(p_net).to(self.config.device) if p_net is not None else None
            self.s_net = torch.nn.DataParallel(s_net).to(self.config.device) if s_net is not None else None
            # self.optim = get_Adam_optim(config, self.net.module)
        else:
            self.net = net.to(self.config.device)
            self.p_net = p_net.to(self.config.device) if p_net is not None else None
            self.s_net = s_net.to(self.config.device) if s_net is not None else None
            # self.optim = get_Adam_optim(config, self.net)
        self.early_stop = config.early_stop
        self.best_dev_acc = 0
        self.unimproved_iters = 0
        self.iters_not_improved = 0

    def pretrain(self):
        self.config.max_epoch = 1
        for epoch in range(0, self.config.max_epoch):
            self.net.train()
            train_loss, train_token_senti, train_token_pred, train_polarity = self.pretrain_epoch()
            epoch_log = ("    Epoch:{:>2}    ".format(epoch)).center(88, "-") + "".center(70, " ") + '\n' + \
                   "total_loss:{:>2.3f}\ttrain_ts:{:>2.3f}\ttrain_tp:{:>2.3f}\ttrain_polarity:{:>2.3f}"\
                       .format(train_loss, train_token_senti, train_token_pred, train_polarity) + "\n"
            print("\r" + epoch_log)

        state = self.net.module.state_dict() if self.config.n_gpu > 1 else self.net.state_dict()
        torch.save(state, self.config.pretrained_path)

        stop_logs = "Pretrained model is saved as: " + self.config.pretrained_path
        print(stop_logs)
            # evaluating phase
            # self.net.eval()
            # with torch.no_grad():
            #     eval_loss, eval_acc, eval_rmse = self.pretrain_eval(self.test_itr)

    def s_train(self):
        # load pretrained vector
        self.load_pretrained()
        for epoch in range(0, self.config.max_epoch):
            self.p_net.train() if self.config.fine_tune else self.p_net.eval()
            self.s_net.train()
            train_loss, train_acc = self.train_epoch()

            logs = ("    Epoch:{:>2}    ".format(epoch)).center(88, "-") + "".center(70, " ") + '\n' + \
                   self.get_logging(train_loss, train_acc, eval="training")
            print("\r" + logs)

            # logging training logs
            # self.logging(self.log_file, logs)

            # evaluating phase
            self.net.eval()
            with torch.no_grad():
                eval_loss, eval_acc = self.s_eval(self.test_itr)
            eval_logs = self.get_logging(eval_loss, eval_acc, eval="evaluating")
            print("\r" + eval_logs)

            # logging evaluating logs
            # self.logging(self.log_file, eval_logs)

            # early stopping
            if eval_acc > self.best_dev_acc:
                self.unimproved_iters = 0
                self.best_dev_acc = eval_acc

                # saving models
                ## getting state
                # state = self.generating_state()
                # torch.save(
                #     state,
                #     self.config.ckpts_path + '/ckpt_{}_{}.pkl'.format(self.config.dataset, self.config.version)
                # )
            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.config.patience and self.early_stop == True:
                    early_stop_logs = "Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, self.best_dev_acc)
                    print(early_stop_logs)
                    # self.logging(self.log_file, early_stop_logs)
                    break

    def load_pretrained(self):
        print("==loading pretrained vector...")
        state = torch.load(self.config.pretrained_path)
        self.p_net.load_state_dict(state, False)
        print("Done!")

    def get_logging(self, loss, acc, eval='training'):
        logs = \
            '==={} phrase...'.format(eval) + "".center(60, " ") + "\n" + \
            "total_loss:{:>2.3f}\ttotal_acc:{:>2.3f}".format(loss, acc) + "\n"
        return logs

    def train_epoch(self):
        acc_fn = multi_acc
        total_loss = []
        total_acc = []
        for step, batch in enumerate(self.train_itr):
            self.optim.zero_grad()
            input_ids, labels = batch
            # print("-"*20)
            # print(input_ids)
            # print(labels)
            # print("-"*20)
            # time.sleep(2)
            input_ids = input_ids.to(self.config.device)
            attention_mask = (input_ids != 0).long().to(self.config.device)  # id of [PAD] is 0
            labels = labels.long().to(self.config.device)
            if self.config.n_gpu > 1:
                embed = self.p_net.module.generate_word_representation(input_ids)
            else:
                embed = self.p_net.generate_word_representation(input_ids)
            if not self.config.fine_tune:
                embed = embed.detach()
            logits, loss = self.s_net(embed, mask=attention_mask, label =labels)

            if self.config.n_gpu > 1: loss = loss.mean()

            metric_acc = acc_fn(labels, logits)

            total_loss.append(loss.item())
            total_acc.append(metric_acc.item())
            loss.backward()
            self.optim.step()

            # monitoring results on every steps
            print(
                "\rIteration: {:>4}/{} ({:>4.1f}%) -- Loss: {:.5f}".format(
                    step, int(len(self.train_itr.dataset) / self.config.batch_size),
                    100 * (step) / int(len(self.train_itr.dataset) / self.config.batch_size),
                    loss),
                end="")
        return np.array(total_loss).mean(), np.array(total_acc).mean()

    def s_eval(self, eval_itr):
        acc_fn = multi_acc
        total_loss = []
        total_acc = []
        for step, batch in enumerate(eval_itr):
            self.optim.zero_grad()
            input_ids, labels = batch
            input_ids = input_ids.to(self.config.device)
            attention_mask = (input_ids != 0).long().to(self.config.device)  # id of [PAD] is 0
            labels = labels.long().to(self.config.device)
            if self.config.n_gpu > 1:
                embed = self.p_net.module.generate_word_representation(input_ids)
            else:
                embed = self.p_net.generate_word_representation(input_ids)
            if not self.config.fine_tune:
                embed = embed.detach()
            logits, loss = self.s_net(embed, mask=attention_mask, label=labels)

            metric_acc = acc_fn(labels, logits)

            if self.config.n_gpu > 1: loss = loss.mean()

            total_loss.append(loss.item())
            total_acc.append(metric_acc.item())

            total_loss.append(loss.item())
            total_acc.append(metric_acc.item())

            # monitoring results on every steps1
            print(
                "\rIteration: {:>4}/{} ({:>4.1f}%)".format(
                    step, int(len(eval_itr.dataset) / self.config.batch_size),
                    100 * (step) / int(len(eval_itr.dataset) / self.config.batch_size)),
                end="")

        return np.array(total_loss).mean(0), np.array(total_acc).mean(0)

    def pretrain_epoch(self):
        sentiment_loss = []
        polarity_loss = []
        token_prediction_loss = []
        total_loss = []
        training_steps_per_epoch = len(self.train_itr)
        for step, batch in enumerate(self.train_itr):
            self.optim.zero_grad()
            input_ids, token_senti_labels, labels = batch
            input_ids = input_ids.to(self.config.device)
            token_senti_labels = token_senti_labels.to(self.config.device)
            attention_mask = (input_ids != 0).long().to(self.config.device)  # id of [PAD] is 0
            labels = labels.long().to(self.config.device)
            if self.config.n_gpu >1:
                loss_token_prediction, loss_target_sentiment, loss_polarity =\
                    self.net.module.pretrain(input_ids, attention_mask, token_senti_labels, labels)
            else:
                loss_token_prediction, loss_target_sentiment, loss_polarity =\
                    self.net.pretrain(input_ids, attention_mask, token_senti_labels, labels)

            loss = 0.3*loss_target_sentiment + 0.2*loss_polarity + loss_token_prediction
            total_loss.append(loss.item())
            sentiment_loss.append(loss_target_sentiment.item())
            polarity_loss.append(loss_polarity.item())
            token_prediction_loss.append(loss_token_prediction.item())
            loss.backward()
            self.optim.step()

            print(
                "\rIteration: {:>3}/{:^3} ({:>4.1f}%) --Total Loss: {:.5f} token_senti_loss: {:.5} polarity_loss: {:.5} token_target_loss: {:.5}".format(
                    step, training_steps_per_epoch / self.config.batch_size, step / training_steps_per_epoch /self.config.batch_size, loss, loss_target_sentiment, loss_polarity, loss_token_prediction) + " "*30,
                end="")

        return np.array(total_loss).mean(), np.array(sentiment_loss).mean(),  np.array(token_prediction_loss).mean(), np.array(polarity_loss).mean()


    def pretrain_eval(self, eval_itr):
        pass

    def run(self, run_mode):
        if run_mode == 'pretrain':
            self.pretrain()
        if run_mode == 'train':
            self.s_train()

