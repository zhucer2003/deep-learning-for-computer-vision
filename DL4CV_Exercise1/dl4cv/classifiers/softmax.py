import numpy as np

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
  
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
  
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
  
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in xrange(num_train):
        scores = np.dot(X[i], W) # (1, num_classes) compute scores for each sample
        scores -= np.max(scores)  # (1, num_classes) shift values to avoid numerical instability
        p = np.exp(scores[y[i]]) / np.sum(np.exp(scores)) # scalar \frac{exp(y_i)}{sum_j_num_classes exp(f_y_j)}
        loss += -np.log(p) # scalar add softmax loss for sample i

        for j in xrange(num_classes): # compute dw_y_j and dw_y_i for each sample i
            if j == y[i]:
                dw_y_i = -X[i] + 1.0 / np.sum(np.exp(scores)) * np.exp(scores[j]) * X[i]
                dW[:, j] += dw_y_i # (D,1) add contribution of sample i to gradient
            else:
                dL_w_j = 1.0 / np.sum(np.exp(scores)) * np.exp(scores[j]) * X[i]
                dW[:, j] += dL_w_j

    dW += reg * W  # add regularization to gradients. note it is not dependent of training samples.
    dW /= num_train  # average gradients
    loss /= num_train  # average loss
    loss += 0.5 * reg * np.sum(W * W)  # add regularization to loss
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
  
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = np.dot(X, W)  # (N,C)
    scores -= np.max(scores, axis=1)[:, np.newaxis] # shift values to avoid numerical instability

    p = np.exp(scores[np.arange(num_train), y]) / np.sum(np.exp(scores), axis=1) # (N,)

    loss += np.sum(-np.log(p))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    grad = np.copy(scores) # (N,C)
    grad = np.exp(grad) # (N,C)
    # exp(f_y_j) * \frac{1}{sum_j_num_classes exp(f_y_j)}
    grad = grad * np.dot(1.0 / np.sum(grad, axis=1)[:, np.newaxis], np.ones((num_classes,))[np.newaxis, :]) # (N,C)

    # setup a selection matrix according to which class each pattern belongs to
    select = np.zeros_like(scores) # (N,C)
    select[np.arange(num_train), y] = -1.0

    # compute gradients. leverage the fact that the gradient contributions of each pattern w.r.t to their true class and
    # the other gradients of each pattern are the same but one "term" that we'll compute later.
    dW = np.dot(X.T, grad) # (D,C) dW = (..) * X_i

    # add the missing term of the gradient contribution of each pattern w.r.t to their true class
    dW += np.dot(X.T, select)

    dW += reg * W  # gradient of regularization loss
    dW /= num_train  # average gradient over all samples
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

