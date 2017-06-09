from __future__ import division
import numpy as np

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param) for the backward pass
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    # extract all relevant variables
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    # spatial output of activation map can be derived from above variables
    H_p = 1 + (H + 2 * pad - HH) / stride
    W_p = 1 + (W + 2 * pad - WW) / stride
    # for the moment assert necessary conditions
    assert int(H_p) == H_p, "H' not an integer"
    assert int(W_p) == W_p, "W' not an integer"

    H_p, W_p = int(H_p), int(W_p) # safe to cast after asserts
    out = np.zeros(shape=(N, F, H_p, W_p), dtype=x.dtype) # allocate known memory for out
    # specify (n_before, n_after) for rows and cols for each dimension
    # e.g. (1,1) specifies to add one row above, one row below and one column before and one column after
    npad = ((0, 0), (1, 1), (1, 1))

    for n in xrange(N): # for each sample
        inp = np.pad(x[n, :, :, :], pad_width=npad, mode='constant', constant_values=0)
        for f in xrange(F): # for each filter
            # go through each element in the f-th activation map
            for hp in xrange(H_p):
                for wp in xrange(W_p):
                    # print('hp %i, wp %i'%(hp,wp))
                    # print('hp*stride:HH+hp*stride %i:%i'%(hp*stride, HH+hp*stride))
                    # print('wp*stride:WW+wp*stride %i:%i' % (wp*stride, WW+wp*stride))
                    out[n, f, hp, wp] = \
                        np.sum(inp[:, (hp*stride):(HH+hp*stride), (wp*stride):(WW+wp*stride)] * w[f, :, :, :]) + b[f]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    # spatial output of activation map can be derived from above variables
    H_p = 1 + (H + 2 * pad - HH) / stride
    W_p = 1 + (W + 2 * pad - WW) / stride

    H_p, W_p = int(H_p), int(W_p)  # safe to cast after asserts

    npad = ((0, 0), (1, 1), (1, 1))

    # backprop on b
    db = np.zeros_like(b)
    for n in range(N):
        for f in range(F):
            for k in range(H_p):
                for l in range(W_p):
                    db[f] += dout[n, f, k, l]

    # backprop on w
    dw = np.zeros_like(w)
    for n in range(N):
        inp = np.pad(x[n, :, :, :], pad_width=npad, mode='constant', constant_values=0)
        for f in range(F):
            for c in range(C):
                for i in range(HH):
                    for j in range(WW):
                        for k in range(H_p):
                            for l in range(W_p):
                                dw[f, c, i, j] += dout[n, f, k, l] * inp[c, i + stride*k, j + stride*l]

    # backprop on x
    dx = np.zeros_like(x)
    for nprime in range(N):
        for cprime in range(C):
            for i in range(H):
                for j in range(W):
                    for f in range(F):
                        for k in range(H_p):
                            for l in range(W_p):
                                for p in range(HH):
                                    for q in range(WW):
                                        if (p + stride*k) == (i + pad) and (q + stride*l) == (j + pad):
                                            dx[nprime, cprime, i, j] += dout[nprime, f, k, l] * w[f, cprime, p, q]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param) for the backward pass
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    N, C, H, W = x.shape
    Hp, Wp, S = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H2, W2 = (H - Hp) / S + 1, (W - Wp) / S + 1
    assert int(H2) == H2, 'H1 not an integer'
    assert int(W2) == W2, 'H2 not an integer'
    H2, W2 = int(H2), int(W2)

    out = np.zeros(shape=(N, C, H2, W2))
    for n in range(N):
        for c in range(C):
            for h2 in range(H2):
                for w2 in range(W2):
                    out[n, c, h2, w2] = np.max(x[n, c, h2*S:h2*S+Hp, w2*S:w2*S+Wp])
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    Hp, Wp, S = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H2, W2 = (H - Hp) / S + 1, (W - Wp) / S + 1
    H2, W2 = int(H2), int(W2)
    dx = np.zeros_like(x)

    for nprime in range(N):
        for cprime in range(C):
            for i in range(H):
                for j in range(W):
                    for k in range(H2):
                        for l in range(W2):
                            x_pooling = x[nprime, cprime, k*S:k*S+Hp, l*S:l*S+Wp]
                            x_max = np.max(x_pooling)
                            x_mask = x[nprime, cprime, :, :] == x_max
                            p_max, q_max = np.unravel_index(x_mask.argmax(), x_mask.shape)
                            if i == p_max and j == q_max:
                                dx[nprime, cprime, i, j] += dout[nprime, cprime, k, l]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    sample_mean = np.mean(x, axis=0)

    x_minus_mean = x - sample_mean

    sq = x_minus_mean**2

    var = 1./N * np.sum(sq, axis=0)

    sqrtvar = np.sqrt(var + eps)

    ivar = 1./sqrtvar

    x_norm = x_minus_mean * ivar

    gammax = gamma * x_norm

    out = gammax + beta

    running_var  = momentum * running_var + (1 - momentum) * var
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean

    cache = (out, x_norm, beta, gamma, x_minus_mean, ivar, sqrtvar, var, eps)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x = (x - running_mean) / np.sqrt(running_var)
    out = x * gamma + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  N, D = dout.shape
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
