import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    
    执行带动量的随机梯度下降。

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value. 一个介于0到1的数值，表明了动量的数值
      Setting momentum = 0 reduces to sgd. 标记动量=0
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients. 速度：一个numpy arry，形状是w， dw被用来存储一个平均
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    # 实现动量更新公式。存储更新的值再next_w变量里。你应该也使用并更新速度v
    
    ####???????这个下面写的很懵逼
    mu = config.get('momentum') #mu= 0.9

#     print("mu=", mu)
    learning_rate = config.get('learning_rate') #learning_rate= 0.01
#     print("learning_rate=", learning_rate)

    # Momentum update rule
    v = mu*v - learning_rate*dw
    w += v
    
    next_w = w
    ###########################################################################
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config['velocity'] = v

    return next_w, config



def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    learning_rate = config.get('learning_rate') #拿到学习率
    decay_rate = config.get('decay_rate') #拿到β
    epsilon = config.get('epsilon')
    cache = config.get('cache') #二阶导数的cache
    
    e = decay_rate*cache + (1-decay_rate)* (dw**2)
    next_w = w - learning_rate / (np.sqrt(e+epsilon)) * dw #(np.sqrt(e+epsilon)) * dw
    
    config['cache'] = e
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    
    使用Adam更新法则，它既包含了梯度平方的移动平均，也有一个偏置校正

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number. #迭代次数，不断加1
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.        
    #
    # 使用Adam更新公式，保存w的下一个值在next_w变量里。记得同时更新m,v,t变量在config里
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################
    
    beta1 = config.get('beta1')
    beta2 = config.get('beta2')
    learning_rate = config.get('learning_rate')
    epsilon = config.get('epsilon')
    m = config.get('m')
    v = config.get('v')
    t = config.get('t') + 1
    
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    m_modify = m / (1 - (beta1 ** t))
    v_modify = v / (1 - (beta2 ** t))
    
    next_w = w - learning_rate / (np.sqrt(v_modify + epsilon)) * m_modify
    
    config['m'] = m
    config['v'] = v
    config['t'] = t
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
