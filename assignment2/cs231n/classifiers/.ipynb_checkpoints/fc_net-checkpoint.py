from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


import random

def gauss_2d(mu, sigma, N, M):
        x = np.random.normal(mu, sigma, N)
        y = np.random.normal(mu, sigma, M)
        return np.column_stack((x,y))

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
    
    注意到这个类没有实现梯度下降；相反，它会和一个独立的Solver的对象进行交互，它负责跑最优化的过程。

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    
    学习的参数存储在字典self.params
    """

    
    
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        
        
#         b1 = np.zeros(input_dim)
#         W1 = weight_scale * np.random.randn(input_dim, hidden_dim)
#         b2 = np.zeros(num_classes)
#         W2 = weight_scale * np.random.randn(hidden_dim, num_classes)
#         self.params['W1'] = W1
#         self.params['W2'] = W2
#         self.params['b1'] = b1
#         self.params['b2'] = b2
        #上面的参数的初始化是不对的，坑死！
        self.params['W1'] = np.random.normal(0, weight_scale, [input_dim, hidden_dim])
        self.params['b1'] = np.zeros([hidden_dim])
        self.params['W2'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
        self.params['b2'] = np.zeros([num_classes])
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        计算loss和梯度对于一个minibatch的数据
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        # 实现两层网络的前向传播，计算网络对X的scores，并把它存储在scores变量里。
        scores_fc1, cache_fc1 = affine_forward(X, self.params['W1'], self.params['b1']) #计算线性的forward的结果，保存在out里
        socres_relu, cache_relu = relu_forward(scores_fc1)
        scores_fc2, cache_fc2 = affine_forward(socres_relu, self.params['W2'], self.params['b2'])
        
        scores = scores_fc2
#         print(scores)
        
        ############################################################################
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                  #
        # 实现两层网络的反向传播。结果保存在字典里。计算数据使用softmax，确保grads[k]保存了self.params[k]，不要忘记加上L2正则项
        # 
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        #下面的无非就是计算loss和grads。loss的计算就是算一个在scores上的softmax的损失
#         dx, dw, db = affine_relu_backward(dout, cache)
        loss, dsoftmax = softmax_loss(scores, y)
        loss += 0.5*self.reg*(np.sum(self.params['W1']*self.params['W1'])+np.sum(self.params['W2']*self.params['W2']))
#         print("loss====", loss)
    
        dx2, dw2, db2 = affine_backward(dsoftmax, cache_fc2)
        drelu = relu_backward(dx2, cache_relu)
        dx1, dw1, db1 = affine_backward(drelu, cache_fc1)

        grads['W2'], grads['b2'] = dw2 + self.reg*self.params['W2'], db2
        grads['W1'], grads['b1'] = dw1 + self.reg*self.params['W1'], db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    
    一个有着任意多层的全连接的神经网络。可以选择实现dropout和batch/layer normlization。对于一个有L层的网络，它的结构将会是：
    
    {线性层 - [batch/layer norm] - relu层 - [dropout层]} 重复出现L-1次 - 线性层 - softmax层
    
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet. 实现一个新的全连接的网络

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer. #这个是一个很重要的参数，代表的是每一个隐藏层的大小
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
          dtype: 
          dtype是一个numpy的数据对象；所有的计算都会用这种类型，float32类型会更快、但是不那么准确，对于数值的梯度检查，你需要使用float64类型
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1 #如果dropout不为1，那么就认为是使用dropout;
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims) #num_layer的个数
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros. 
        # 当使用batch normalization的时候，保存scale、改变第一层参数的gamma1和beta1, 对于第二层网络使用gamma2和beta2，等等。
        # 尺度参数(scale)应当初始值为1，
        ############################################################################
#         self.params['W1'] = np.random.normal(0, weight_scale, [input_dim, hidden_dim])
#         self.params['b1'] = np.zeros([hidden_dim])
#         self.params['W2'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
#         self.params['b2'] = np.zeros([num_classes])
#         print("num_classes", num_classes)
#         print("input_dim=", input_dim)
#         print("self.num_layers=", self.num_layers)
        for i in range(self.num_layers):
            if (i == 0):
                last_dim = input_dim
            else:
                last_dim = hidden_dims[i-1]
            
            if (i == self.num_layers-1):
                next_dim = num_classes
            else:
                next_dim = hidden_dims[i]
            
            if self.normalization=='batchnorm':
                self.params['beta' + str(i+1)] = np.zeros([hidden_dims[i]])
                self.params['gamma' + str(i+1)] = np.ones([hidden_dims[i]])
                
            self.params['W'+str(i+1)] = np.random.normal(0, weight_scale, [last_dim, next_dim])
            self.params['b'+str(i+1)] = np.zeros(next_dim)
        
        
        '''
        print("num_classes", num_classes)
        print("input_dim=", input_dim)
        for i in range(self.num_layers - 1):
            self.params['W' + str(i+1)] = np.random.normal(0, weight_scale, [input_dim, hidden_dims[i]])
            self.params['b' + str(i+1)] = np.zeros([hidden_dims[i]])

            if self.normalization=='batchnorm':
                self.params['beta' + str(i+1)] = np.zeros([hidden_dims[i]])
                self.params['gamma' + str(i+1)] = np.ones([hidden_dims[i]])

            input_dim = hidden_dims[i]  # Set the input dim of next layer to be output dim of current layer.

        # Initialise the weights and biases for final FC layer
        self.params['W' + str(self.num_layers)] = np.random.normal(0, weight_scale, [input_dim, num_classes])
        print("SSSape", self.params['W'+str(self.num_layers)].shape)
        self.params['b' + str(self.num_layers)] = np.zeros([num_classes])
        '''

                                                     
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        # 当使用dropout的时候，我们需要传递一个dropout_param的字典给每一个dropout层，因此这个层知道dropout概率和模式(训练集/测试集)
        # 你需要传递相同的dropout_param给每一个dropout层
        self.dropout_param = {}
        if self.use_dropout: 
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        # 对于batch normalization, 我们需要关注运行的平均值和方差，所以我们需要传递一个特定的bn_param对象来给每一个batch normlization层。
        # 你应当传递self.bn_params[0]给第一层的batch normalization层
        self.bn_params = []
        if self.normalization=='batchnorm': #!!!self.normlization有两种方式，一种是batch-normalization, 另外一种则是layernorm
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
#         np_model = np.array([])
        cache_fc = {}#[np_model for i in range(self.num_layers)]
        cache_relu = {}#[np_model for i in range(self.num_layers)]
        cache_dropout = {}#[np_model for i in range(self.num_layers)]
        
        for i in range(self.num_layers - 1): #最后一层是不一样的，因为只有一个单单的affine, 因此需要差别对待
            scores_fc, cache_fc[i] = affine_forward(X, self.params['W'+str(i+1)], self.params['b'+str(i+1)]) #计算线性的forward的结果，保存在out里
#             print("fc shape", scores_fc.shape)
            if self.normalization=='batchnorm':
                scores_bn, cache_bn = batchnorm_backward(scores_fc, self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i]) #gamma, beta, bn_param):
                scores_relu, cache_relu[i] = relu_forward(scores_bn) 
            else:
                scores_relu, cache_relu[i] = relu_forward(scores_fc)
#                 print("@@@", i, cache_relu[i].shape)
#             print("Relu shape:", scores_relu.shape)
           
            X = scores_relu #相当于滚动给下一个循环的X使用
            if self.use_dropout:
                scores_dropout, cache_dropout[i] = dropout_forward(scores_relu, self.dropout_param) #突然有个疑惑，为什么所有层的dropout的参数是一样的，而每一层的bn_params都不一样(连变量名都加上s了）
                X = scores_dropout
#         print("Before last shape", X.shape)    
#         print("param shape", self.params['W'+str(self.num_layers)].shape)
        scores, final_cache = affine_forward(X, self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])  
#         print(scores.shape)
        
                            
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

#         print("-----------------------------------------")
        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #先把好算的loss给算了
        loss, dsoftmax = softmax_loss(scores, y)
        loss += 0.5*self.reg*(np.sum(self.params[str('W'+str(self.num_layers))]*self.params[str('W'+str(self.num_layers))])) #???为什么这个正则化的误差只要计算最后一层的权重了？之前的那些层不要了吗
#         print(loss)
        
        ###然后反向传播计算grads了
#         dx2, dw2, db2 = affine_backward(dsoftmax, cache_fc2)
#         drelu = relu_backward(dx2, cache_relu)
#         dx1, dw1, db1 = affine_backward(drelu, cache_fc1)

#         grads['W2'], grads['b2'] = dw2 + self.reg*self.params['W2'], db2
#         grads['W1'], grads['b1'] = dw1 + self.reg*self.params['W1'], db1
        
        
        dx_final, dw_final, db_final = affine_backward(dsoftmax, final_cache)
        grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] = dw_final + self.reg*self.params['W'+str(self.num_layers)], db_final
        
        dx_last = dx_final
#         print("dx_last.shape=", dx_last.shape)
        for i in range(self.num_layers - 1, 0, -1):
#             print("i=", i)
#             print("cache_relu.shape=", cache_relu[i-1].shape)
            if self.use_dropout: #如果再relu层之后有一个dropout，我们需要在dropout层后加上这个
                ddropout = dropout_backward(dx_last, cache_dropout[i-1])
                dx_last = ddropout
            drelu = relu_backward(dx_last, cache_relu[i-1])
            dx, dw, db = affine_backward(drelu, cache_fc[i-1])
            dx_last = dx
            grads['W'+str(i)], grads['b'+str(i)] = dw, db
            
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
