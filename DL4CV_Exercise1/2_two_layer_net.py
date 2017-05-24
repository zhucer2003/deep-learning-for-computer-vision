
import numpy as np
import matplotlib.pyplot as plt

from dl4cv.classifiers.neural_net import TwoLayerNet

from dl4cv.data_utils import load_CIFAR10
from dl4cv.vis_utils import visualize_cifar10

def get_CIFAR10_data(num_training=48000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. 
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/'
    X, y = load_CIFAR10(cifar10_dir)
    

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

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    return X, y, X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

def train_model(X_train, y_train, X_val, y_val):
    best_net = None # store the best model into this
    best_stats = None
    results = {}
    best_val = -1
    learning_rates = [1e-3] # 1e-3 [1.9e-3 1e-3, 1e-6, 1e-9]
    momentums = [0.5] # [0.5, 0.9, 0.95, 0.99]
    regularization_strengths = [1e-5]
    hidden_sizes = [400]
    iters = 10000
    batch_sizes = [512]
    inp_size = 32 * 32 * 3
    num_classes = 10
    #################################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best trained  #
    # model in best_net.                                                            #
    #                                                                               #
    # To help debug your network, it may help to use visualizations similar to the  #
    # ones we used above; these visualizations will have significant qualitative    #
    # differences from the ones we saw above for the poorly tuned network.          #
    #                                                                               #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
    # write code to sweep through possible combinations of hyperparameters          #
    # automatically like we did on the previous exercises.                          #
    #################################################################################
    for lr in learning_rates:
        for reg in regularization_strengths:
            for hs in hidden_sizes:
                for bs in batch_sizes:
                    for mom in momentums:
                        print("lr, reg, hs, bs, mom", lr, reg, hs, bs, mom)
                        net = TwoLayerNet(input_size=inp_size, hidden_size=hs, weight_init='normal',
                                          momentum=mom, output_size=num_classes)

                        # Train the network
                        stats = net.train(X_train, y_train, X_val, y_val,
                            num_iters=iters, batch_size=bs,
                            learning_rate=lr, learning_rate_decay=0.95,
                            reg=reg, verbose=False)

                        # Predict on the validation set
                        train_acc = (net.predict(X_train) == y_train).mean()
                        val_acc = (net.predict(X_val) == y_val).mean()

                        results[(lr, reg, hs, bs, mom)] = (train_acc, val_acc)
                        if val_acc > best_val:
                            best_val = val_acc
                            best_net = net
                            best_stats = stats
    #################################################################################
    #                               END OF YOUR CODE                                #
    #################################################################################
    # Print out results.
    for lr, reg, hs, bs, mom in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg, hs, bs, mom)]
        print 'lr %e reg %e hidden %e batch-size %e mom %e train accuracy: %f val accuracy: %f' % (
                    lr, reg, hs, bs, mom, train_accuracy, val_accuracy)

    print 'best validation accuracy achieved during cross-validation: %f' % best_val
    return best_net


def test_model(best_net):
    test_acc = (best_net.predict(X_test) == y_test).mean()
    print 'Test accuracy: ', test_acc

def save_model(best_net):
    from dl4cv.model_savers import save_two_layer_net
    save_two_layer_net(best_net)
    print("Saved")

if __name__ == '__main__':
    X, y, X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()

    # Invoke the above function to get our data.
    X_raw, y_raw, X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
    print 'Train data shape: ', X_train.shape
    print 'Train labels shape: ', y_train.shape
    print 'Validation data shape: ', X_val.shape
    print 'Validation labels shape: ', y_val.shape
    print 'Test data shape: ', X_test.shape
    print 'Test labels shape: ', y_test.shape
    print 'dev data shape: ', X_dev.shape
    print 'dev labels shape: ', y_dev.shape

    best_net = train_model(X_train, y_train, X_val, y_val)
    test_model(best_net)
    save_model(best_net)
