from __future__ import print_function, division
from builtins import range
import numpy as np

"""
This file defines layer types that are commonly used for recurrent neural
networks.

这个文件定义了常在RNN神经网络中使用的层次的类型
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    #print("x=", x)
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.
    
    执行每一个时间戳的vanilla RNN，使用的是tanh激活层。

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    
    输入数据有D维，隐藏层有H维，我们使用的是一个为N的minibatch大小的数据。

    Inputs:
    - x: Input data for this timestep, of shape (N, D). 对于这个时间戳的隐藏层的数据，形状是(N,D)
    - prev_h: Hidden state from previous timestep, of shape (N, H) 前一个时间戳的隐藏层的状态，形状是(N,H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H) input-to-hidden的权重矩阵
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H) hidden-hidden的权重矩阵
    - b: Biases of shape (H,) 偏置

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H) 下一个隐藏层的状态
    - cache: Tuple of values needed for the backward pass. 需要进行反向传播的cache
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    # 执行一个单个前向传播的步骤，使用vanilla RNN。保存下一个隐藏层的状态和任何你反向传播需要用到的值在next_h
    # 和cache里
    ##############################################################################
    next_h = np.tanh(prev_h.dot(Wh) + x.dot(Wx) + b) #(N,H)
    #print(next_h) #(N,H)
    #print(next_h.shape) 
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    cache = x, prev_h, Wx, Wh, b, next_h
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.
    对于vanilla RNN某一时间戳的反向传播

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H) 对于下一个隐藏层的loss的梯度
    - cache: Cache object from the forward pass
    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D) dx
    - dprev_h: Gradients of previous hidden state, of shape (N, H) 前向状态的梯度
    - dWx: Gradients of input-to-hidden weights, of shape (D, H) dWx inpu-to-hidden层的权重的梯度
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H) hidden-to-hidden层的梯度
    - db: Gradients of bias vector, of shape (H,) 偏置层的梯度
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    # 实现RNN的单步的反向传播                                                       #
    # 提示：对于tanh函数，你可以计算局部的导数在
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    
    x, prev_h, Wx, Wh, b, next_h = cache #(N,H) #目前呢，我是只保存了这个在cache里呢
    dx = (dnext_h*(1 - next_h * next_h)).dot(Wx.T)
    #print(dx.shape)
    dprev_h = (dnext_h*(1-next_h * next_h)).dot(Wh.T) #为什么这个计算的不对呢？#后来改对了，改成转置就对了（这个Wh是一个H*H的，要转置这个并不能配出来）
    dWx = x.T.dot( dnext_h*(1 - next_h * next_h) )
    dWh = prev_h.T.dot( dnext_h*(1 - next_h * next_h)  )
    db = np.sum(dnext_h*(1 - next_h * next_h), axis = 0)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.
    
    执行一个Vaniila的RNN前向传播在整个数据上。我们认为一个输入序列包含了T个向量，每个向量的维度都是D。
    RNN使用一个隐藏层大小是H，并且我们在一个包含了N个序列的minibatch上工作。在执行RNN的前向传播后，我们
    返回对于所有时间戳的隐藏层的状态。

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D). N个minibatch，T个时间戳， D维度
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N, T, D = x.shape
    _, H = h0.shape
    prev_h = h0
    h = np.zeros((N, T, H))
    cache = {}
    for i in range(T):
        next_h, cache_i = rnn_step_forward(x[:, i, :], prev_h, Wx, Wh, b)
        h[:,i,:] = next_h
        prev_h = next_h
        cache[i] = cache_i
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.
    
    计算整个序列数据的反向传播

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    输入是上游的所有隐藏层状态的梯度，形状是（N，T，H）
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).
    
    注意： dh包含了每一个时间戳产生的单独的损失函数的梯度，而不是

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N, T, H = dh.shape
    _, D = cache[0][0].shape
    dWx = np.zeros((D,H))
    dWh = np.zeros((H,H))
    db = np.zeros((H,))
    dx = np.zeros((N,T,D))
    
    dnext_h = np.zeros((N,H))
    for i in range(T-1,-1,-1):
        #x, prev_h, Wx, Wh, b, next_h = cache[i] #这个其实用不着啦
        dnext_h += dh[:,i,:]
        dx_i, dprev_h, dWx_i, dWh_i, db_i = rnn_step_backward(dnext_h, cache[i])
        dWx += dWx_i
        dWh += dWh_i
        db += db_i
        dnext_h = dprev_h
        dx[:,i,:] = dx_i
    dh0 = dprev_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.
    
    对于word embeddings的前向传播。我们在一个大小为N的minibatch（每一个序列长度为T）执行
    我们考虑一个词典有V个单词，每个单词有维度D。

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V. 
    - W: Weight matrix of shape (V, D) giving word vectors for all words.
        对于每个单词的权重
      
     x:
     [[0 3 1 2]
     [2 1 0 3]]
     
     有两个句子，序列分别是（单词0，单词3，单词1，单词2）；（单词2，单词1，单词0，单词3）
     

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    #print(W)
    out, cache = None, None
    #print("------------------------------")
    #print(x)
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    # 
    ##############################################################################
    out = W[x, :]
    cache = x, W
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D) #上游的梯度，形状是(N,T,D)
    - cache: Values from the forward pass #从前传播传过来的cache数据

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    word embedding矩阵的word embedding的梯度
    """
    dW = None
    
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, W = cache
    N, T, D = dout.shape
    print("N,T,D=", N,T,D)
    V, _ = W.shape
    dW = np.zeros((V, D))
    #print("x=")
    #print(x)
    #print("W=")
    #print(W)
    #print("dout=")
    #print(dout)
    np.add.at(dW, x, dout)
    # 哇擦，这个np.add.at真他妈的饶。总算理清楚了
    # 首先，这个x的形状是 N*T, dout的形状是N*T*D。然后x里的值的范围是[0,V]，这个很重要！
    # 然后我们在这个的反向传播的意图就是，给这些x对应的词加上后面传过来的梯度。所以就是对于x里的每一个值，增加上dout对应的值
    # （官方API里说的can broadcast就是这个道理啦，其实并不饶了）
    #print("------dW = ")
    #print(dW)
    #dw = W.T
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.
    
    对于一个时间的线性层的前向传播。输入是一个D维的向量，被赋值为一个有N个时间戳的向量，每一个都是长度T。
    
    我们使用一个线性层来转化这些向量到一个新的有M维的向量。
    
    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.
    
    我们人为我们正在对于一个大小为V的词典做预测。输入x给了所有词典元素的所有时间戳，y给了在每一个时间戳的正确的元素。我们使用交叉熵计算所有时间戳的误差，并求平均值。

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.
    
    作为一个复杂，我们也许希望忽略在某些时间戳上的输出，因为不同的长度也许被加到同一个minibatch并padded用NULL。
    
    可选的mask参数告诉我们哪些元素希望对loss进行贡献

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss. #布尔数组，形状是(N,T) mask(i,t)告诉了在x[i,t]的元素是否要加入到loss里

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N # 直接用mask_flat一乘就直接过滤掉那些我们不希望传播的
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
