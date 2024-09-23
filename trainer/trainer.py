import datetime
import logging
import os
import random

import numpy as np
import torch
from torch import nn

import data_loader
from model import build_model


class Trainer():
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
        self.train_loader, self.test_loader = self.set_dataloader(args)

    def set_seed(self, seed):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def set_cuda(self, use_cuda=False, gpu_id='0'):
        if use_cuda and torch.cuda.is_available():
            device = torch.device('cuda:' + gpu_id)
            print("Train with GPU：{}".format(torch.cuda.get_device_name()))
        else:
            device = torch.device('cpu')
            print("Train with CPU")
        return device

    def set_logger(self, args):
        # logs dir
        current_time = datetime.datetime.now()
        file_name = current_time.strftime("%Y-%m-%d_%H-%M-%S.log")
        folder_path = os.path.join(args.log_root, args.model, args.mode)
        if args.target != '':
            folder_path = os.path.join(folder_path, args.target)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        logs_path = os.path.join(folder_path, file_name)
        # create logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # file
        fh = logging.FileHandler(filename=logs_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        # console
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        print('logs path: {}'.format(logs_path))
        if not os.path.exists(logs_path):
            raise Exception('create logs failed!')
        return logs_path, logger

    def set_model_path(self, args):
        # model weights(.pth) dir
        model_dir = os.path.join(args.model_root, args.model, args.mode)
        if args.target != '':
            model_dir = os.path.join(model_dir, args.target)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print('model path: {}'.format(model_dir))
        return model_dir

    def set_dataloader(self, args):
        if args.mode == 'mixture':
            # mix source and target data
            train_loader = data_loader.load_training_mixture(args.img_size, args.dataset_root, args.source, args.target,
                                                             args.batch_size)
            test_loader = data_loader.load_testing(args.img_size, args.dataset_root, args.target, args.batch_size)
        elif args.mode == 'target_only':
            # train only with target data
            train_loader = data_loader.load_training(args.img_size, args.dataset_root, args.target, args.batch_size)
            test_loader = data_loader.load_testing(args.img_size, args.dataset_root, args.target, args.batch_size)

        len_train_dataset = len(train_loader.dataset)
        len_test_dataset = len(test_loader.dataset)
        len_train_loader = len(train_loader)
        len_test_loader = len(test_loader)

        print('Number of Train Dataset : {}'.format(len_train_dataset))
        print('Number of Train Loader : {}'.format(len_train_loader))
        print('Number of Test Dataset : {}'.format(len_test_dataset))
        print('Number of Test Loader : {}'.format(len_test_loader))

        return train_loader, test_loader

    def load_pretrained_weights(self, args):
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=self.device)
            del_keys = ['head.weight', 'head.bias'] if self.model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]

            print(self.model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in self.model.named_parameters():
                if "head" not in name and "pre_logits" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

    def train_one_epoch(self, epoch):
        len_train_loader = len(self.train_loader)
        len_train_data = len(self.train_loader.dataset)
        self.model.train()
        # loss
        criterion = nn.CrossEntropyLoss()
        # optim
        pg = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(pg, lr=self.learning_rate, momentum=0.9, weight_decay=5E-5)
        # iter
        train_loader_iter = iter(self.train_loader)
        i = 1
        while i <= len_train_loader:
            images, labels = next(train_loader_iter)
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % self.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, i * len(images), len_train_data,
                    100. * i / len_train_loader, loss.item()))
            i = i + 1

    def test(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def train(self, args):
        # print train parameters
        self.logger.info(args)
        # Staring Train
        old_model_path = ''
        best_acc = 0
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
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
