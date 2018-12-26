from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:、
    
    一个三层的卷积神经网络
    
    形状是：卷积层-relu层-2*2的最大池化层-线性层-relu层-线性层-softmax层

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    这个网络在数据上执行一个minibatch，形状是（N,C,H,W)
    
    包括N张图片、每一个有高度H、宽度W和C个channel
        
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
#         对于三层的卷积神经网络来初始化权重和偏置
#         权重的初始化是一个均值在0、标准差是weight_scale的高斯分布；偏置应当初始化为0
#         所有的权重和偏执应当存在字典self.params中
#         保存卷积层的权重和偏置在W1, b1
#         使用W2和b2对于隐藏层的线性层
#         保存W3和b3在输出的线性层中

        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        
        C, H, W = input_dim

        self.params['W1'] = np.random.normal(0, weight_scale, [num_filters, C, filter_size, filter_size]) #卷积层的初始化权重 #后面那两维什么情况?
        self.params['b1'] = np.zeros([num_filters])
        self.params['W2'] = np.random.normal(0, weight_scale, [np.int(H/2)*np.int(H/2)*num_filters, hidden_dim])
        self.params['b2'] = np.zeros([hidden_dim])
        self.params['W3'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
        self.params['b3'] = np.zeros([num_classes])
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        
        计算这个三层网络的loss和梯度了

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        scores_conv, cache_conv = conv_forward_fast(X, self.params['W1'], self.params['b1'], conv_param) #conv_forward_naive(X, self.params['W1'], self.params['b1'], conv_param) #计算卷积层的forward的结果，保存在out里
        socres_relu1, cache_relu1 = relu_forward(scores_conv)
        scores_maxpool, cache_maxpool = max_pool_forward_naive(socres_relu1, pool_param)
        scores_fc1, cache_fc1 = affine_forward(scores_maxpool, self.params['W2'], self.params['b2'])
        scores_relu2, cache_relu2 = relu_forward(scores_fc1)
        scores_fc2, cache_fc2 = affine_forward(scores_relu2, self.params['W3'], self.params['b3'])
        
        scores = scores_fc2
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        #下面的无非就是计算loss和grads。loss的计算就是算一个在scores上的softmax的损失
#         dx, dw, db = affine_relu_backward(dout, cache)
        loss, dsoftmax = softmax_loss(scores, y)
        loss += 0.5*self.reg*(np.sum(self.params['W1']*self.params['W1'])+np.sum(self.params['W2']*self.params['W2'])+np.sum(self.params['W3']*self.params['W3']))
#         print("loss====", loss)
    
        dx_fc2, dw_fc2, db_fc2 = affine_backward(dsoftmax, cache_fc2)
        drelu2 = relu_backward(dx_fc2, cache_relu2)
        dx_fc1, dw_fc1, db_fc1 = affine_backward(drelu2, cache_fc1)
        dx_maxpool = max_pool_backward_naive(dx_fc1, cache_maxpool)
        drelu1 = relu_backward(dx_maxpool, cache_relu1)
        dx_conv, dw_conv, db_conv = conv_backward_fast(drelu1, cache_conv) #conv_backward_naive(drelu1, cache_conv)

        grads['W3'], grads['b3'] = dw_fc2 + self.reg*self.params['W3'], db_fc2
        grads['W2'], grads['b2'] = dw_fc1 + self.reg*self.params['W2'], db_fc1
        grads['W1'], grads['b1'] = dw_conv + self.reg*self.params['W1'], db_conv
        
        
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
