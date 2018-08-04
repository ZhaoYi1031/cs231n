import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  结构化的SVM代价函数，简单实现（有循环的）
  Structured SVM loss function, naive implementation (with loops).

  输入有D维，C个种类，并且我们在N个样本中执行minibatch
  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights. #10*3073维
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  # 接下来就是计算SVM损失（这个定义可以看课程的视频，简单而言，就是如果W*X计算后的得分+1大于正确标签的得分（这个就相当于预测错了，即怎么可能比真实的标签值的得分还要大），那么就加上这个偏差，否则无视（我们可以这么理解，对于小的显然是应该的，所以就不应该加到损失函数里，否则就是加上偏差）
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[:, j] += X[i, :] #错误的标签的权重应当减少 #因为之后的SGD上有一个-，因此在这儿是减号 #至于为什么导数是X[i], 是因为由loss函数我们可以直观地看到导数项 #比如说j是5，即第五个分类，此时我们将权重矩阵中第5列的所有参数(3073个值)都加上X[i]这一行的值
        dW[:, y[i]] -= X[i, :] #正确的标签的参数应当增加
        loss += margin
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train #除掉训练集的个数
  dW /= num_train
  dW += reg*W

  # Add regularization to the loss.
  loss += 0.5*reg * np.sum(W * W) #加上正则项的惩罚，在这里我们采用的是L2范式，L2范式通过对每一个参数进行平方后进行相加，通过这样的方式来抑制参数不要太大
  # 有些地方好像写的是0.5*reg*np.sum(W*W) #我看有个知乎评论是这样的：这个应该是看regulation loss 如何定义，通常加入0.5是为了计算regulation loss的梯度时系数变为1

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  #接下来是要计算损失函数的梯度并把它存储到dW里，按照这段的说法是我们直接在上面计算偏导数!

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  W: 3073*10
  X: 500*3073
  y: 500,
  reg: 常数，0.000005
  """

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores = X.dot(W) # shape: 500*10, 即每一个样本对于每一个种类都有一个得分
  correct_class_scores = scores[np.arange(num_train), y] #shape: 500,
  correct_class_scores = np.reshape(correct_class_scores, (-1, 1))
  margin = scores - correct_class_scores + 1 #(500,10)
  margin[np.arange(num_train), y] = 0 #把正确标签的偏差值置为0
  margin[margin < 0] = 0
  loss = np.sum(margin)
  loss /= num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin[margin>0] = 1
  one_num_per_line = np.sum(margin, axis = 1) #统计每一行里面的1的和（其实也就是1的个数）
  margin[np.arange(num_train), y] = -one_num_per_line
  dW = X.T.dot(margin)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
