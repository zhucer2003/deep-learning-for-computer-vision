import time
import random
import numpy as np
from dl4cv.model_savers import save_softmax_classifier
import matplotlib.pyplot as plt
from dl4cv.data_utils import load_CIFAR10

def load_dataset():
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/'
    X, y = load_CIFAR10(cifar10_dir)
    return X,y


def split_dataset(X, y):
    # Split the data into train, val, and test sets. In addition we will
    # create a small development set as a subset of the data set;
    # we can use this for development so our code runs faster.
    num_training = 48000
    num_validation = 1000
    num_test = 1000
    num_dev = 500

    assert (num_training + num_validation + num_test) == 50000, 'You have not provided a valid data split.'

    # Our training set will be the first num_train points from the original
    # training set.
    mask = range(num_training)
    X_train = X[mask]
    y_train = y[mask]

    # Our validation set will be num_validation points from the original
    # training set.
    mask = range(num_training, num_training + num_validation)
    X_val = X[mask]
    y_val = y[mask]

    # We use a small subset of the training set as our test set.
    mask = range(num_training + num_validation, num_training + num_validation + num_test)
    X_test = X[mask]
    y_test = y[mask]

    # We will also make a development set, which is a small subset of
    # the training set. This way the development cycle is faster.
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # In[ ]:

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # As a sanity check, print out the shapes of the data
    print 'Training data shape: ', X_train.shape
    print 'Validation data shape: ', X_val.shape
    print 'Test data shape: ', X_test.shape
    print 'dev data shape: ', X_dev.shape

    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev


def preprocess_data(X_train, X_val, X_test, X_dev):
    # Preprocessing: subtract the mean image
    # first: compute the image mean based on the training data
    mean_image = np.mean(X_train, axis=0)

    # second: subtract the mean image from train and test data
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image


    # third: append the bias dimension of ones (i.e. bias trick) so that our classifier
    # only has to worry about optimizing a single weight matrix W.
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    print X_train.shape, X_val.shape, X_test.shape, X_dev.shape
    return X_train, X_val, X_test, X_dev


def train_model(X_train, y_train, X_val, y_val):
    from dl4cv.classifiers import Softmax

    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the fraction
    # of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    learning_rates = [2.37e-06]
    batch_sizes = [8162]
    momentums = [0.95]
    regularization_strengths = [1e-1] # 6.36e+05, 1e-1

    ################################################################################
    # TODO:                                                                        #
    # Write code that chooses the best hyperparameters by tuning on the validation #
    # set. For each combination of hyperparameters, train a classifier on the      #
    # training set, compute its accuracy on the training and validation sets, and  #
    # store these numbers in the results dictionary. In addition, store the best   #
    # validation accuracy in best_val and the Softmax object that achieves this    #
    # accuracy in best_softmax.                                                    #
    #                                                                              #
    # Hint: You should use a small value for num_iters as you develop your         #
    # validation code so that the classifiers don't take much time to train;       #
    # once you are confident that your validation code works, you should rerun     #
    # the validation code with a larger value for num_iters.                       #
    ################################################################################
    from numpy.random import random_sample

    max_count = 10

    for lr in learning_rates:
        for reg in regularization_strengths:
            for mom in momentums:
                for bs in batch_sizes:
                    softmax = Softmax()
                    softmax.train(X_train, y_train, learning_rate=lr, reg=reg, momentum=mom,
                                  num_iters=2500,batch_size=bs, verbose=False)
                    y_train_pred = softmax.predict(X_train)
                    y_val_pred = softmax.predict(X_val)
                    tr_acc = np.mean(y_train == y_train_pred)
                    val_acc = np.mean(y_val == y_val_pred)
                    results[(lr, reg, mom, bs)] = (tr_acc, val_acc)
                    if val_acc > best_val:
                        best_val = val_acc
                        best_softmax = softmax
    ################################################################################
    #                              END OF YOUR CODE                                #
    ################################################################################

    # Print out results.
    for lr, reg, mom, bs in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg, mom, bs)]
        print 'lr %e reg %e mom %e bs %e train accuracy: %f val accuracy: %f' % (
                    lr, reg, mom, bs, train_accuracy, val_accuracy)

    print 'best validation accuracy achieved during validation: %f' % best_val
    return best_softmax


def test_model(X_test, y_test, best_softmax):
    y_test_pred = best_softmax.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print 'softmax on raw pixels final test set accuracy: %f' % (test_accuracy, )


def save_model(best_softmax):
    from dl4cv.model_savers import save_softmax_classifier
    save_softmax_classifier(best_softmax)


if __name__ == '__main__':
    X,y = load_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = split_dataset(X, y)
    X_train, X_val, X_test, X_dev = preprocess_data(X_train, X_val, X_test, X_dev)
    best_softmax = train_model(X_train, y_train, X_val, y_val)
    test_model(X_test, y_test, best_softmax)
    best_softmax = save_model(best_softmax)
    print("Finished")
