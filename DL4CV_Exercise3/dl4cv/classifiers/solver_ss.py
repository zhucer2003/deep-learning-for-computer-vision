from random import shuffle
import numpy as np
import time
import torch.nn as nn
import copy
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={}, num_classes=23):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.num_classes = num_classes

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.epoch_loss = []

    def train(self, model, dataset_loader, num_epochs=10, log_nth=100):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """

        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        t = 0
        num_iterations = len(dataset_loader['train']) * num_epochs
        # give background zero weight, all other classes get equal weight of one !
        weight = torch.ones(self.num_classes)
        weight[0] = 0

        print 'START TRAIN.'
        ############################################################################
        # TODO:                                                                    #
        # Write your own personal training method for our solver. In Each epoch    #
        # iter_per_epoch shuffled training batches are processed. The loss for     #
        # each batch is stored in self.train_loss_history. Every log_nth iteration #
        # the loss is logged. After one epoch the training accuracy of the last    #
        # mini batch is logged and stored in self.train_acc_history.               #
        # We validate at the end of each epoch, log the result and store the       #
        # accuracy of the entire validation set in self.val_acc_history.           #
        #
        # Your logging should look something like:                                 #
        #   ...                                                                    #
        #   [Iteration 700/4800] TRAIN loss: 1.452                                 #
        #   [Iteration 800/4800] TRAIN loss: 1.409                                 #
        #   [Iteration 900/4800] TRAIN loss: 1.374                                 #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                                #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                                #
        #   ...                                                                    #
        ############################################################################
        start_time = time.time()
        # iterate over epochs
        for epoch in xrange(num_epochs):

            epoch_loss = 0.0

            # iterate first over training phase
            for phase in ['train']:
                # don't train model during validation !
                if phase == 'train':
                    model.train(True)
                else:
                    model.train(False)


                # iterate over the corresponding data in each phase
                for data in dataset_loader[phase]:

                    inputs, labels = data

                    inputs = Variable(inputs)
                    labels = Variable(labels)
                    outputs = model(inputs)

                    # set gradients to zero for each mini_batch iteration !
                    optim.zero_grad()

                    loss = self.wrap(outputs, labels, weight=weight, pixel_average=True)
                    epoch_loss += loss.data[0]
                    if phase == 'train':
                        if t % log_nth == 0:
                            print '[Iteration %d / %d] TRAIN loss: %f' % \
                                  (t + 1, num_iterations, loss.data[0])
                        t += 1
                        loss.backward()
                        optim.step()
                print('[Epoch %d / %d] TRAIN acc: %f' % (epoch + 1, num_epochs, epoch_loss))
        print('Trained in {0} seconds.'.format(int(time.time() - start_time)))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        print 'FINISH.'

    def wrap(self, inputs, targets, weight=None, pixel_average=True):
        n, c, h, w = inputs.size()

        # after this we have n, h, w, c = inputs.size()
        inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
        # expand targets to a tensor with depth of c
        inputs = inputs[targets.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

        # also exclude background and unlabeled
        targets_mask = targets >= 0
        targets = targets[targets_mask]

        loss = F.cross_entropy(inputs, targets, weight=weight, size_average=False)
        if pixel_average:
            loss /= targets_mask.data.sum()
        return loss