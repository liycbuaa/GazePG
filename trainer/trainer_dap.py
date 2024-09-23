import os

import numpy as np
import torch
from torch.autograd import Variable

import data_loader
from model import build_model
from model.RevGrad import ReverseLayerF
from trainer.trainer import Trainer


class TrainerDAP(Trainer):
    def __init__(self, args):
        self.set_seed(args.seed)
        self.device = self.set_cuda(args.cuda, args.gpu_id)
        self.learning_rate = args.learning_rate
        self.log_interval = args.log_interval
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size

        self.model = build_model(args).to(self.device)
        print(self.model)
        # load pretrained weights
        if args.weights != "":
            self.load_pretrained_weights(args)

        self.logs_path, self.logger = self.set_logger(args)
        self.model_path = self.set_model_path(args)
        self.train_source_loader, self.train_target_loader, self.test_loader = self.set_dataloader(args)

    def set_dataloader(self, args):
        train_source_loader = data_loader.load_training(args.img_size, args.dataset_root, args.source, args.batch_size)
        train_target_loader = data_loader.load_training(args.img_size, args.dataset_root, args.target, args.batch_size)
        test_loader = data_loader.load_testing(args.img_size, args.dataset_root, args.target, args.batch_size)

        len_source_dataset = len(train_source_loader.dataset)
        len_target_dataset = len(train_target_loader.dataset)
        len_test_dataset = len(test_loader.dataset)

        print('Number of Train Source Dataset : {}'.format(len_source_dataset))
        print('Number of Train Target Dataset : {}'.format(len_target_dataset))
        print('Number of Test Dataset : {}'.format(len_test_dataset))

        return train_source_loader, train_target_loader, test_loader

    def train_one_epoch(self, epoch):
        len_source_loader = len(self.train_source_loader)
        len_target_loader = len(self.train_target_loader)
        len_train_data = len(self.train_source_loader.dataset)
        self.model.train()
        # loss
        loss_class = torch.nn.CrossEntropyLoss()
        loss_domain = torch.nn.CrossEntropyLoss()
        # optim
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # iter
        data_source_iter = iter(self.train_source_loader)
        data_target_iter = iter(self.train_target_loader)
        dlabel_src = Variable(torch.ones(self.batch_size).long().to(self.device))
        dlabel_tgt = Variable(torch.zeros(self.batch_size).long().to(self.device))
        i = 1
        while i <= len_source_loader:
            # the parameter for reversing gradients
            p = float(i + epoch * len_source_loader) / self.num_epochs / len_source_loader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # for the source domain batch
            source_data, source_label = next(data_source_iter)
            source_data, source_label = Variable(source_data.to(self.device)), Variable(source_label.to(self.device))

            # feature, class_output, domain_output
            emb_src, clabel_src, dlabel_pred_src = self.model(source_data, alpha=alpha)
            label_loss = loss_class(clabel_src, source_label)
            domain_loss_src = loss_domain(dlabel_pred_src, dlabel_src)

            # for the target domain batch
            target_data, target_label = next(data_target_iter)
            # 109200 % 520 = 0
            if i % len_target_loader == 0:
                data_target_iter = iter(self.train_target_loader)
            target_data, target_label = Variable(target_data.to(self.device)), Variable(target_label.to(self.device))
            emb_tgt, clabel_tgt, dlabel_pred_tgt = self.model(target_data, alpha=alpha)
            # target loss
            label_loss = label_loss + loss_class(clabel_tgt, target_label)

            domain_loss_tgt = loss_domain(dlabel_pred_tgt, dlabel_tgt)

            alpha = 2.0
            clip_thr = 0.3
            mix_weight = 1.0
            # feature-level mixup
            mix_ratio = np.random.beta(alpha, alpha)
            mix_ratio = round(mix_ratio, 2)
            # clip the mixup ratio
            if (mix_ratio >= 0.5 and mix_ratio < (0.5 + clip_thr)):
                mix_ratio = 0.5 + clip_thr
            if (mix_ratio > (0.5 - clip_thr) and mix_ratio < 0.5):
                mix_ratio = 0.5 - clip_thr

            dlabel_mix = Variable((torch.ones(self.num_epochs) * mix_ratio).long().to(self.device))
            emb_mix = mix_ratio * emb_src + (1 - mix_ratio) * emb_tgt
            reverse_emb_mix = ReverseLayerF.apply(emb_mix, alpha)
            dlabel_pred_mix = self.model.domain_classifier(reverse_emb_mix)
            domain_loss_mix = loss_domain(dlabel_pred_mix, dlabel_mix)

            domain_loss_total = domain_loss_src + domain_loss_tgt + domain_loss_mix * mix_weight
            loss_total = label_loss + domain_loss_total

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if i % self.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlabel_Loss: {:.6f}\tdomain_Loss: {:.6f}'.format(
                    epoch + 1, i * len(source_data), len_train_data,
                    100. * i / len_source_loader, label_loss.item(), domain_loss_total.item()))
            i = i + 1

    def test(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                _, outputs, _ = self.model(images, alpha=0)
                pred = outputs.max(1)[1]
                total += labels.size(0)
                correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
        return correct / total

    def train(self, args):
        # print train parameters
        self.logger.info(args)
        # Staring Train
        old_model_path = ''
        best_acc = 0
        for epoch in range(self.num_epochs):
            # train
            self.train_one_epoch(epoch)
            # validate
            new_acc = self.test()
            if new_acc > best_acc:
                best_acc = new_acc
                if old_model_path != '':
                    os.remove(old_model_path)
                new_model_file = 'best_model_' + '{:.2f}'.format(
                    100. * best_acc) + '.pkl'
                new_model_path = os.path.join(self.model_path, new_model_file)
                torch.save(self.model.state_dict(), new_model_path)
                old_model_path = new_model_path
            self.logger.info(
                'Epoch[{}/{}]\tAccuracy: {:.2f}%\tBest Accuracy: {:.2f}%'.format(epoch + 1, self.num_epochs,
                                                                                 100 * new_acc,
                                                                                 100 * best_acc))
        self.logger.info('===============Training Over===============')
        return old_model_path
