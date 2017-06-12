import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ThreeLayerCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride=1, weight_scale=0.001, pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride: Only for convolutional layer
        - weight_scale: Scale for the convolution weights initialization-
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ThreeLayerCNN, self).__init__()
        channels, height, width = input_dim

        ############################################################################
        # TODO: Initialize the necessary layers to resemble the ThreeLayerCNN      #
        # architecture  from the class docstring. In- and output features should   #
        # not be hard coded which demands some calculations especially for the     #
        # input of the first fully convolutional layer. The convolution should use #
        # "some" padding which can be derived from the kernel size and its weights #
        # should be scaled. Layers should have a bias if possible.                 #
        ############################################################################
        # define parameters
        # if stride = 1 try to preserve same input size after conv layer,
        # otherwise we don't care if input gets scaled down
        pad = 0
        if stride == 1:
            pad = (kernel_size - 1) / 2
        conv_out_width = 1 + (width - kernel_size + 2 * pad) / stride
        conv_out_height = 1 + (height - kernel_size + 2 * pad) / stride
        out_pool_width = 1 + (conv_out_width - pool) / pool
        out_pool_height = 1 + (conv_out_height - pool) / pool
        # input features of first fc layer
        lin_input = num_filters * out_pool_height * out_pool_width
        # this way we can easily access them in forward/backward pass
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_scale = weight_scale
        self.pool = pool
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=channels, out_channels=num_filters, kernel_size=kernel_size,
        #               stride=stride, padding=pad, bias=True),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=pool)
        # )
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=lin_input, out_features=hidden_dim, bias=True),
        #     nn.Dropout(),
        #     nn.ReLU(),
        #     nn.Linear(in_features=hidden_dim, out_features=num_classes, bias=True)
        # )

        # self.conv = nn.Conv2d(in_channels=channels, out_channels=num_filters, kernel_size=kernel_size,
        #                       stride=stride, padding=pad, bias=True)
        # self.conv_relu = nn.ReLU(),
        # self.pool = nn.MaxPool2d(kernel_size=pool)
        # self.fc = nn.Linear(in_features=lin_input, out_features=hidden_dim, bias=True)
        # self.fc_dropout = nn.Dropout()
        # self.fc2_relu = nn.ReLU()
        # self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes, bias=True)

        self.conv = nn.Conv2d(in_channels=channels, out_channels=num_filters, kernel_size=kernel_size,
                              stride=stride, padding=pad, bias=True)
        # self.conv.weight.data = weight_scale * self.conv.weight.data # weight scale
        init.xavier_normal(self.conv.weight, gain=np.sqrt(2))
        init.constant(self.conv.bias, 0.001)
        self.fc1 = nn.Linear(in_features=lin_input, out_features=hidden_dim, bias=True)
        init.xavier_normal(self.fc1.weight, gain=np.sqrt(2))
        init.constant(self.fc1.bias, 0.001)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes, bias=True)
        init.xavier_normal(self.fc2.weight, gain=np.sqrt(2))
        init.constant(self.fc2.bias, 0.001)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ############################################################################
        # TODO: Chain our previously initialized convolutional neural network      #
        # layers to resemble the architecture drafted in the class docstring.      #
        # Have a look at the Variable.view function to make the transition from    #
        # convolutional to fully connected layers.                                 #
        ############################################################################
        # print('before conv', x.data.size())
        x = self.conv(x)
        # print('after conv', x.data.size())
        x = F.relu(F.max_pool2d(x, kernel_size=self.pool))
        # print('after max pool', x.size())
        (_, C, H, W) = x.data.size()
        x = x.view(-1, C * H * W)
        # print('after view', x.data.size())
        x = F.relu(F.dropout(self.fc1(x), self.dropout))
        # print('after fc1', x.size())
        x = self.fc2(x)
        # print('after fc2', x.size())
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print 'Saving model... %s' % path
        torch.save(self, path)
