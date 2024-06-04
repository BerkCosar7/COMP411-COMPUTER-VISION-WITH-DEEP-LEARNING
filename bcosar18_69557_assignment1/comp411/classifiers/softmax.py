import numpy as np
from random import shuffle
import builtins

def softmax_loss_naive(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg_l2: (float) regularization strength for L2 regularization
    - reg_l1: (float) default: 0. regularization strength for L1 regularization 
                to be used in Elastic Net Reg. if supplied, this function uses Elastic
                Net Regularization.

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0.:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]


    for i in range(num_train):
        scores = X[i].dot(W)
        shift_scores = scores - np.max(scores)  # for numerical stability
        loss_i = -shift_scores[y[i]] + np.log(np.sum(np.exp(shift_scores)))
        loss += loss_i

        for j in range(num_classes):
            softmax_output = np.exp(shift_scores[j]) / np.sum(np.exp(shift_scores))
            if j == y[i]:
                dW[:, j] += (-1 + softmax_output) * X[i]
            else:
                dW[:, j] += softmax_output * X[i]

    loss /= num_train
    dW /= num_train

    # Regularization
    if regtype == 'L2':
        loss += 0.5 * reg_l2 * np.sum(W * W)
        dW += reg_l2 * W
    else:  # ElasticNet, i.e., L1 and L2 combined
        

        loss += reg_l1 * np.sum(np.abs(W))  # L1 part
        dW += reg_l1 * np.sign(W)  # L1 gradient part

        loss += 0.5 * reg_l2 * np.sum(W * W)  # L2 part
        dW += reg_l2 * W  # L2 gradient part

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.   #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
   
    scores = X.dot(W)
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    
   
    softmax_probs = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores), axis=1, keepdims=True)
    
    
    correct_class_probs = softmax_probs[np.arange(X.shape[0]), y]
    loss = -np.sum(np.log(correct_class_probs)) / X.shape[0]
    
    
    softmax_probs[np.arange(X.shape[0]), y] -= 1
    dW = X.T.dot(softmax_probs) / X.shape[0]
    
    
    if regtype == 'L2':
        loss += 0.5 * reg_l2 * np.sum(W * W)
        dW += reg_l2 * W
    else:  
        loss += 0.5 * reg_l2 * np.sum(W * W) + reg_l1 * np.sum(np.abs(W))
        dW += reg_l2 * W + reg_l1 * np.sign(W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
