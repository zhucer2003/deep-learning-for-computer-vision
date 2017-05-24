from dl4cv.data_utils import load_CIFAR10
from dl4cv.features import hog_feature, color_histogram_hsv, extract_features
from dl4cv.model_savers import save_feature_neural_net
from dl4cv.classifiers.neural_net import TwoLayerNet
import numpy as np

def get_CIFAR10_data(num_training=48000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for feature extraction and training.
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

    return X, y, X_train, y_train, X_val, y_val, X_test, y_test


def extract_feats(X_train, X_val, X_test):

    num_color_bins = 10 # Number of bins in the color histogram
    feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
    X_train_feats = extract_features(X_train, feature_fns, verbose=True)
    X_val_feats = extract_features(X_val, feature_fns)
    X_test_feats = extract_features(X_test, feature_fns)

    # Preprocessing: Subtract the mean feature
    mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
    X_train_feats -= mean_feat
    X_val_feats -= mean_feat
    X_test_feats -= mean_feat

    # Preprocessing: Divide by standard deviation. This ensures that each feature
    # has roughly the same scale.
    std_feat = np.std(X_train_feats, axis=0, keepdims=True)
    X_train_feats /= std_feat
    X_val_feats /= std_feat
    X_test_feats /= std_feat

    # Preprocessing: Add a bias dimension
    X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
    X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
    X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

    return X_train_feats, X_test_feats, X_val_feats


def train_model(X_train_feats, y_train, X_val_feats, y_val):


    input_dim = X_train_feats.shape[1]
    num_classes = 10
    learning_rates = [6.0e-1] # 6.0e-1 [1e-1, 1.5e-1, 2.5e-1, 3e-1, 6e-1]
    regularization_strengths = [1e-4] # 1e-6, 1e-7, 1e-8
    momentums = [0.5]
    hidden_sizes = [300]
    iters = 15000
    batch_sizes = [2048]

    best_net = None # store the best model into this
    best_stats = None
    results = {}
    best_val = -1
    ################################################################################
    # TODO: Train a two-layer neural network on image features. You may want to    #
    # validate various parameters as in previous sections. Store your best   #
    # model in the best_net variable.                                              #
    ################################################################################
    for lr in learning_rates:
        for reg in regularization_strengths:
            for hs in hidden_sizes:
                for bs in batch_sizes:
                    for mom in momentums:
                        net = TwoLayerNet(input_size=input_dim, hidden_size=hs, weight_init='normal',
                                                  momentum=mom, output_size=num_classes)

                        # Train the network
                        stats = net.train(X_train_feats, y_train, X_val_feats, y_val,
                            num_iters=iters, batch_size=bs,
                            learning_rate=lr, learning_rate_decay=0.95,
                            reg=reg, verbose=False)

                        # Predict on the validation set
                        train_acc = (net.predict(X_train_feats) == y_train).mean()
                        val_acc = (net.predict(X_val_feats) == y_val).mean()

                        results[(lr, reg, hs, bs, mom)] = (train_acc, val_acc)
                        if val_acc > best_val:
                            best_val = val_acc
                            best_net = net
                            best_stats = stats
    ################################################################################
    #                              END OF YOUR CODE                                #
    ################################################################################
    # Print out results.
    for lr, reg, hs, bs, mom in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg, hs, bs, mom)]
        print 'lr %e reg %e hidden %e bs %e mom %e train accuracy: %f val accuracy: %f' % (
                    lr, reg, hs, bs, mom, train_accuracy, val_accuracy)

    print 'best validation accuracy achieved during cross-validation: %f' % best_val
    return best_net

def test_model(net):
    test_acc = (net.predict(X_test_feats) == y_test).mean()
    print test_acc


def save_model(best_net):
    save_feature_neural_net(best_net)
    print("Saved")


if __name__ == '__main__':
    X, y, X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

    # Invoke the above function to get our data.
    X_raw, y_raw, X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
    print 'Train data shape: ', X_train.shape
    print 'Train labels shape: ', y_train.shape
    print 'Validation data shape: ', X_val.shape
    print 'Validation labels shape: ', y_val.shape
    print 'Test data shape: ', X_test.shape
    print 'Test labels shape: ', y_test.shape

    X_train_feats,X_test_feats, X_val_feats = extract_feats(X_train, X_val, X_test)

    # sanity checks
    print 'Train data shape: ', X_train_feats.shape
    print 'Train labels shape: ', y_train.shape
    print 'Validation data shape: ', X_val_feats.shape
    print 'Validation labels shape: ', y_val.shape
    print 'Test data shape: ', X_test_feats.shape
    print 'Test labels shape: ', y_test.shape

    best_net = train_model(X_train_feats, y_train, X_val_feats, y_val)
    test_model(best_net)
    save_model(best_net)