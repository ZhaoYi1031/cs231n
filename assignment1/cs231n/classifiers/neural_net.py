from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    #权重被初始化成小的随机值，截距初始化为0，权重和截距存储在self.params变量里，它是一个字典有如下的几个key: W1是第一层的参数，shape是(D,H); b1是第一层的参数，shape是(H,)；W2是第二层的参数，shape是（H,C) b2是第二层权重参数，shape是(C,)
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size) 
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    # 计算损失函数和梯度对于一个有两个连接层的全连接网路。
    # 输入是X: shape是(N, D); 每一个X[i]都是一个训练数据样例。
    # y[i]是训练集x[i]所代表的label。参数y是一个可选参数，如果它没有传递啊那么我们只需要返回得分；否则我们就也要返回梯度。
    # reg: 正则化参数
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1'] #从参数字典中拿到数据
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None #最后应该是每一个每一个数据X[i]对于种类C有一个得分score
    
    C = W2.shape[1]
    scores = np.zeros((N, C))
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    scores_fully_1 = X.dot(W1) + b1 #(N,D) * (D,H) = (N,H)
    #print("scores_fully_1:",scores_fully_1)
    scores_relu = np.maximum(scores_fully_1, 0) #(N,H)
    #print("scores_relu:",scores_relu)
    scores_fully_2 = scores_relu.dot(W2) + b2 #(N,H) * (H,C) = (N,C)
    #print("scores_fully_2:",scores_fully_2)
    
    scores = scores_fully_2 
    #!!!一直以来我以为那个scores是在softmax层之后的得分，实则不然，我们的softmax层是来求loss的。注意scores和loss的区别
    #喂，softmax是一个求一个求损失的函数，不是激活层，艹能不能搞清楚！
        
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    
    #下面就是计算loss的过程了
    loss = 0 #要初始化一下，不然会出现"TypeError: unsupported operand type(s) for +: 'NoneType' and 'float'"的错误
    #print(y)
    scores -= np.max(scores) #实际上这个会降低那么一点点误差精度 1e-7级别
    
    ppp = np.zeros(N)
    
    for i in range(N): #对于每一个数据去操作
        #print("i=", i)
        
        loss_exp = np.exp(scores[i]) #先把所有的求一个幂指
        #print(loss_exp)
        tot = np.sum(loss_exp)
        #print("tot=", tot)
        #print("val=", np.exp(scores[i][y[i]]) / tot)
        #print("val=", -np.log(np.exp(scores[i][y[i]]) / tot))
        if (tot != 0):
            loss = loss - np.log((np.exp(scores[i][y[i]])) / tot)
#         ppp[i] = - np.log((np.exp(scores[i][y[i]])) / tot)
    #np.mean(ppp)
    loss = loss/N + 0.5*reg*(np.sum(W1 * W1) + np.sum(W2 * W2)) #!!!有个除n，被坑了好久 ###这里没有0.5*reg, 否则误差是0.01有点大
#     print("loss=", loss)
    #print("loss = ", loss)
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    #下面就是动人心弦地求梯度的过程了！！！反向传播一下！！！即从后往前一步步往前去推
    #主要难点还是用到之前的求softmax的导数的过程没注意梯度下降的主要思路就是
    #先计算一下W2的梯度好了
    
    dL_df = np.zeros((N,C))
    for i in range(N):
        loss_exp = np.exp(scores[i])
        tot = np.sum(loss_exp)
        for j in range(C):
            dL_df[i,j] = loss_exp[j] / tot
            if (j == y[i]):
                dL_df[i,j] = dL_df[i,j] - 1
#     dL_df /= N ###!!!为什么这个也要除n???
#     print(dL_df)
#     print("------------------------------------------------------------------")
   
    dW2 = np.dot(scores_relu.T, dL_df)
    
    grads['W2'] = dW2 
    grads['W2'] /= N
    grads['W2'] += reg * W2 
    
    db2 = np.sum(dL_df, axis=0) #这个我其实有点懵!!! axis=0是求每一列的和，所以就是把(N,C)压缩到了(C,)
    grads['b2'] = db2
    grads['b2'] /= N
#     print(grads['W2'])
#     print("------------------------------------------------------------------")
    
    dr_dfc1 = np.zeros(scores_relu.shape)
    dW1_1 = dL_df.dot(W2.T) #(N,C) * (C,H) = (N,H)
    for i in range(dW1_1.shape[0]):
        for j in range(dW1_1.shape[1]):
            if (scores_relu[i][j] == 0):
                dW1_1[i][j] = 0
#                 dr_dfc1[i][j] = 1
#     dW1_1 = dW1_1*dr_dfc1
    
    dW1 = X.T.dot(dW1_1) #(D,N) * (N,H) = (D,H)
    grads['W1'] = dW1 
    grads['W1'] /= N
    grads['W1'] += reg * W1
        
     
    db1 = np.sum(dW1_1, axis=0)
    grads['b1'] = db1 
    grads['b1'] /= N
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads
 
  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

#     print(num_train)
#     print(batch_size)
#     print(iterations_per_epoch)
#     print(num_iters)
    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      
      idxs = np.random.choice(num_train, batch_size, replace=True) #这个的num_train为5，而batch_size为100，倘若我们
#       print(idxs)
      X_batch = X[idxs]
#       print(X_batch)
      y_batch = y[idxs]
#       print(y_batch)
      
    
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
#       print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
#       print(loss)
#       print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#       print(grads)
      loss_history.append(loss)
#       print("*******************************")

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      ###使用我们计算的梯度的列表来更新我们的网络的梯度，存储在self.params里的
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['b2'] -= learning_rate * grads['b2']
#       print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      # 每一个epoch，检查train和val的准确度和衰减系数
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    N = X.shape[0]
    C = self.params['b2'].shape[0]
    y_pred = np.zeros(N)
    scores = np.maximum(X.dot(self.params['W1']) + self.params['b1'], 0).dot(self.params['W2']) + self.params['b2']
    
#     print(scores.shape)
#     print(C)
    
    for i in range(N):
        score_max = -1e9
        pos = -1
        for j in range(C):
            if (scores[i][j] > score_max):
                score_max = scores[i][j]
                pos = j
            y_pred[i] = pos
#     print(y_pred)     
#     print("CNMMMMMMMMMMMMMMMMMMMMMMMMM")
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


