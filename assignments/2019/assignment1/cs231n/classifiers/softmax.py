from builtins import range
import numpy as np
import math
from random import shuffle
from past.builtins import xrange

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
    dW = np.zeros_like(W).transpose()
    num_classes = W.shape[1]
    num_train = X.shape[0]    

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_train):
      scores = X[i].dot(W)
      scores -= np.max(scores)


      exps = np.exp(scores)
      denominator  = np.sum(exps)
      numerator = exps[y[i]]

      for j in range(num_classes):
        factor = exps[j] / denominator
        if j == y[i]:
            factor -= 1
        dW[j] += X[i] * factor

      loss_i = -math.log(numerator / denominator)

      loss += loss_i

    loss /= num_train

    loss += reg * np.sum(W * W)              

    dW /= num_train    
    dW = dW.transpose()
    dW += reg * 2 * W              

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W)
    scores_exps = np.exp(scores)
    correct_exps = scores_exps[ np.arange(num_train), y]
    score_sums = np.sum(scores_exps, axis = 1)

    loss_vector = -np.log(correct_exps / score_sums)
    loss = np.sum(loss_vector) / num_train
    loss += reg * np.sum(W * W)              

    factors = scores_exps / np.reshape(score_sums, [num_train, 1])
    factors[np.arange(num_train), y] -= 1

    dW = X.transpose().dot(factors)
    dW /= num_train
    dW += reg * 2 * W 

    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
