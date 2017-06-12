from random import shuffle
import numpy as np
import time

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func()

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=1):
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
        iter_per_epoch = len(train_loader)
        num_iterations = num_epochs * iter_per_epoch
        t = 0
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
        for epoch in xrange(num_epochs):
            for iter, (x_batch, y_batch) in enumerate(train_loader):
                optim.zero_grad()
                x = Variable(x_batch, requires_grad=False)
                y = Variable(y_batch, requires_grad=False)
                x_out = model(x)
                loss = self.loss_func(x_out, y)
                loss.backward()
                optim.step()

                self.train_loss_history.append(loss.data[0])

                if t % log_nth == 0:
                    print '[Iteration %d / %d] TRAIN loss: %f' % \
                          (t + 1, num_iterations, self.train_loss_history[-1])
                t += 1

            y_pred = np.argmax(x_out.data.cpu().numpy(), axis=1)
            pred_acc = np.count_nonzero(y_pred == y.data.cpu().numpy()) / float(len(y_pred))
            self.train_acc_history.append(pred_acc)

            print '[Epoch %d / %d] TRAIN acc/loss: %f/%f' % \
                  (epoch+1, num_epochs, self.train_acc_history[-1], self.train_loss_history[-1])

            y_val_pred = []

            for x_val_batch, y_val_batch in val_loader:
                x_val = Variable(x_val_batch, requires_grad=False)
                y_val = Variable(y_val_batch, requires_grad=False)
                x_val_out = model(x_val)
                val_loss = self.loss_func(x_val_out, y_val)

                y_pred = np.argmax(x_val_out.data.cpu().numpy(), axis=1)
                val_pred_acc = np.count_nonzero(y_pred == y_val.data.cpu().numpy()) / float(len(y_pred))
                y_val_pred.append(val_pred_acc)
            val_pred_acc = np.mean(y_val_pred)
            self.val_acc_history.append(val_pred_acc)

            print '[Epoch %d / %d] VAL acc/loss: %f/%f' % \
                  (epoch + 1, num_epochs, self.val_acc_history[-1], val_loss.data[0])

        print('Trained in {0} seconds.'.format(int(time.time() - start_time)))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        print 'FINISH.'
