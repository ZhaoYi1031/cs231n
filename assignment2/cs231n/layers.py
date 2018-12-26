from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    
    输入x是一个维数很多的array，有形状(N, d_1, ... d_k) 并且包括了一个N个例子的minibatch, 每一个例子的x[i]有形状(d_1, .. d_k).
    我们reshape每个input成一个vector有维度所有的...
    然后转化成一个output vector为维度M。
    

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k) #(2, 4, 5, 6)
    - w: A numpy array of weights, of shape (D, M) #(120, 3)
    - b: A numpy array of biases, of shape (M,) #(3,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
#     print(w.shape)
#     print(x.shape)
#     print("-----------------------------------------")
#     print(x[0][0][0])
#     print("*****************************************")
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    N = x.shape[0]
    M = w.shape[0]
#     print(M)
    x_new = np.reshape(x, (N, -1))
#     x_new = np.zeros((N, M))
#     for i in range(N):
#         x_new[i] = np.reshape(x[i], -1)#把它reshape成一个
    out = x_new.dot(w) + b
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    这个是计算对于已经求好的dout，然后再计算(wx+b)的对于x、w和b的导数，链式法则乘起来就好了
   
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k) #(10, 2, 3)
      - w: Weights, of shape (D, M) #(6, 5)
      - b: Biases, of shape (M,) #(5,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
#     print(x.shape)
#     print(w.shape)
#     print(b.shape)
#     print("----------------------------------------")
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N = x.shape[0]
    db = np.sum(dout, axis = 0) #按照后一层的导数每行的值求个和 #实际上我们可以考虑成为了形状相同的一个操作
    x_temp = np.reshape(x, (N, -1)) #(N, D)
#     print(x_temp.shape)
    dw = x_temp.T.dot(dout)
    dx = dout.dot(w.T) #(N,M) * (M,D)
    dx = dx.reshape(x.shape)
    
    dw = x_temp.T.dot(dout)#(D,N)*(N,M)  = (D,M)
    db = np.sum(dout, axis = 0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    计算relu的前向传播的值

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    计算relu的反向传播的导数
    
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    x_tmp = np.zeros(x.shape)
    x_tmp[x>0] = 1
    dx = dout * x_tmp
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    
    batch normlization的前向传播过程

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.
    
    在训练中，样本的方差和（非正确）的样本的方差从minibatch的数据，并且被用来正规化到来的数据。
    **在训练过程中，我们还保持每个特征的均值和方差的指数衰减运行平均值，并且这些平均值用于在测试时对数据进行归一化**

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    
    在每个时间戳我们用动量规则来更新数据的方差和方差使用一个指数级别的衰减。

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.
    
    注意到BN的paper建议一个不同的测试时间的方法：他们计算样本的均值和方差对于每一个特征，使用一个大量的训练图像即而不是使用一个执行的average.
    对于这个实现，我们选择来使用running average相反，因为他们不要求一个额外的估计层，BN的torch7实现也使用了running average。

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
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        # 实现training time的前向传播对于batch norm。使用minibatch的数据来正规化之后的数据，并规模和转移这个正规化的数据使用gamma和beta参数
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        #  你应当保存输出在变量out中。对于需要的反向传播的任何变量，应当保存在cache变量中
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #  你应当也一起使用计算的样本平均值和变量用momentum变量来更新running mean和running variance，保存结果在running_mean和running_var中
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #  注意到尽管你需要追踪这些running变量，你也应该normalize这些数据基于标准偏差上（这些变量的平均误差）！
        sample_mean = np.mean(x, axis = 0)
#         print('sample_mean =', sample_mean)
        
        sample_var = np.var(x, axis = 0)
#         print('sample_var = ', sample_var)
        
        normalized_data = (x - sample_mean) / np.sqrt(sample_var + eps) #这个是正则化数据，这个好理解 #“scale and shift”操作。为了让因训练所需而“刻意”加入的BN能够有可能还原最初的输入
        out = gamma * normalized_data + beta #out = γ * 正则化数据 + β
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean #如果读者做过滤波，这和低通滤波器类似。每次更新时把之前的值衰减一点点（乘以一个momentum，一般很大，如0.9,0.99），然后把当前的值加一点点进去(1-momentum)。
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
    #         print('running_mean = ', running_mean, 'running_var =', running_var)
        
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
        
        cache = {
            'x_minus_mean': (x - sample_mean),
            'normalized_data': normalized_data,
            'gamma': gamma,
            'ivar': 1./np.sqrt(sample_var + eps),
            'sqrtvar': np.sqrt(sample_var + eps),
        }
            
        #######################################################################
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        
        out = (gamma / (np.sqrt(running_var + eps)) * x) + (beta - (gamma * running_mean)) / np.sqrt(running_var + eps) 
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
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

    要求的是关于x的导数、关于gamma、关于beta的导数

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    N, D = dout.shape
    #print('N = ', N, 'D = ', D)
    normalized_data = cache.get('normalized_data')
    gamma = cache.get('gamma')
    ivar = cache.get('ivar')
    x_minus_mean = cache.get('x_minus_mean')
    sqrtvar = cache.get('sqrtvar')
    
    dbeta = np.sum(dout, axis = 0)
    dgamma = np.sum(dout * normalized_data, axis = 0)
    
    #print("------------")
    #print(ivar.shape)
    #print(ivar)
    #print("------------")
    
    dx_normalized = dout * gamma       # [N,D]
    
    tt = np.sum(dout * gamma * x_minus_mean, axis = 0, keepdims=True)
    #print('tt.shape=', tt.shape)
    #print(tt)
    #print('ivar.shape=', ivar.shape)
    #print(ivar)
    dsample_var = -0.5 * np.sum(dout * gamma * x_minus_mean, axis = 0, keepdims=True) * ivar ** 3#-0.5 * np.sum(dx_normalized * x_minus_mean, axis=0, keepdims=True) * ivar**3
    kk = tt*ivar
    #print(kk)
    #print("dsample_var.shape=", dsample_var.shape)
    
    
    dsample_mean = -1.0 * np.sum(dx_normalized * ivar, axis=0, keepdims=True) - 2.0 * dsample_var * np.mean(x_minus_mean, axis=0, keepdims=True)
    dx1 = dx_normalized * ivar
    dx2 = 2.0/N * dsample_var * x_minus_mean
    dx = dx1 + dx2 + 1.0/N * dsample_mean
    
#     dx = np.zeros((N, D))
    
#     for i in range(N):
#         mul1 = dout[i] * gamma
#         sub1 = (1 - 1.0 / N) * ivar[i]
#         sub2 = x_minus_mean[i] * x_minus_mean[i] / N * ivar[i] * ivar[i] * ivar[i]
#         dx[i] = mul1 * (sub1 - sub2)
#     print(dx.shape)
#     print(dx)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    
    这个就是要求BN的其它的反向传播的方法咯

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
    
    在这个实现中你应当求的是BN的反向传播的导数，尽可能的形式简单。
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    N, D = dout.shape
    normalized_data = cache.get('normalized_data')
    gamma = cache.get('gamma')
    ivar = cache.get('ivar')
    x_minus_mean = cache.get('x_minus_mean')
    sqrtvar = cache.get('sqrtvar')
    
    dbeta = np.sum(dout, axis = 0)
    dgamma = np.sum(dout * normalized_data, axis = 0)
    
    dx =(1 / N) * gamma * ivar * ((N * dout) - np.sum(dout, axis=0) - (x_minus_mean) * np.square(ivar) * np.sum(dout * (x_minus_mean), axis=0))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param): #实现dropout层的前向传播
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
        我们保持的数据是概率为p的
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
        如果模式是train，那么执行dropout；如果模式是test，那么只需要返回input就好了
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        # 对于插入的dropout, 实现训练阶段的前向传播，保存dropout的mask在mask变量里。  #
        #######################################################################
        # During training randomly drop out neurons with probability P, here we create the mask that does this.
        #print("xxxxx=", x)
        mask = (np.random.random_sample(x.shape) >= p)
        #print("mask=",mask)
        
        out = x*mask
        #print("oooooooooout=", out)

        # Inverted dropout scales the remaining neurons during training so we don't have to at test time.
        dropout_scale_factor = 1/(1-p) ###骚操作!!! inverted dropout
        #print("dropout_scale_factor = ", dropout_scale_factor)
        mask = mask*dropout_scale_factor
        #print("mmmmmm=", mask)

        # Apply the dropout mask to the input.
        out = x*mask
        #print("out=", out)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out=x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    
    一个卷积层网络的naive的实现的前向传播

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.
    
    输入层包括N个数据点，每一个点有C个通道，高度为H，宽度是W。我们卷积每个数据用F个不同的filter，在每一个filter中扫描所有的C个通道，并且有高度HH和宽度WW

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        
    输入:
    
    x为输入的数据，形状是（N, C, H, W)
    
    这个信息非常重要，也就是第一维是有N个数据，第二维是channel的个数（一般图片是RGB三个通道），然后F代表有F个不同的滤波器，每一个滤波器也是有C个通道，然后滤波器的大小是HH*WW
    
    w是过滤器的权重，形状是(F, C, HH, WW)
    
    b是偏移，形状是(F, )
    
    卷积层的参数，是一个字典，有如下的一些值：
    
    stride： 像素的数量在不同的相邻的接收范围，在水平和垂直的方向上
    
    pad： 像素的数量来用来0-pad输入（？）
    
    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.
    
    在padding中，pad令应当被放置对称的（例如在每一边是相等的）
    注意不要修改直接原始的输入x

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    
    输出的结果是一个tuple，包括：
    out，是输出的数据，有形状（N, F, H', W')
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.
    # 在这个里面实现卷积层的前向传播
    # Hint: you can use the function np.pad for padding.
    # 提示：你可以使用np.pad来实现padding
    ###########################################################################
    
    pad, stride = conv_param['pad'], conv_param['stride']
    #print(x)
    #print(type(pad))
    #print ('x.shape=', x.shape)
    x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values = (0, )) #这个就是padding的操作啦，所以其实是后面的两维，分别在左右对称的各自加上padding的大小，然后值是0
    #print ('x_padded.shape=', x_padded.shape)
    #print(x_padded)
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    output_height = int(1 + (H + 2 * pad - HH) / stride) #计算最终输出的array的大小，
    output_width = int(1 + (W + 2 * pad - WW) / stride)
    
    #print('output_height = ', output_height)
    out = np.zeros((N, F, output_height, output_width)) #初始化最终的output的大小是N个数据图、F个filter、每个都是一个feature map（即output_height * output_width）
    
    
    for i in range(output_height):
        for j in range(output_width):
            #print('i = ', i, 'j =', j)
            x_padded_mask = x_padded[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW] #取出这一片切片信息出来 #取出HH*WW的input的地方（其中x和y方向都是不断增加stride的大小在不断移动的）
            #print(x_padded_mask.shape,'!!!') #(2, 3, 4, 4)
            
            for k in range(F): #对于每一个过滤器
                #print(w[k, :, :, :].shape, '???') (3, 4, 4)
                val = np.sum(x_padded_mask * w[k, :, :, :], axis = (1,2,3))
                #print(x_padded_mask * w[k, :, :, :])
                #print("np.sum(x_padded_mask * w[%d, :, :, :], axis = (1,2,3)="%k, val)
                out[:, k, i, j] = np.sum(x_padded_mask * w[k, :, :, :], axis = (1,2,3)) #最难理解的地方！其实就是将输入的切片信息和我们的filter相乘（叫点积更为合适一些）#第k个filter、feature_map在坐标(i,j)位置处的计算 #右边的计算出来的是多个filter的结果，然后最后求sum的时候就
    
    out = out + (b)[None, :, None, None]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
    
    卷积层的反向传播的naive版

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # Grab conv parameters and pad x if needed.
    x, w, b, conv_param = cache
    stride = conv_param.get('stride')
    pad = conv_param.get('pad')
    padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant'))

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, H_out, W_out = dout.shape

    # Initialise gradient output tensors.
    dx_temp = np.zeros_like(padded_x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # Calculate dB.
    # Just like in the affine layer we sum up all the incoming gradients for each filters bias.
    for ff in range(F):
        db[ff] += np.sum(dout[:, ff, :, :])

    # Calculate dw.
    # By chain rule dw is dout*x
    for nn in range(N):
        for ff in range(F):
            for jj in range(H_out):
                for ii in range(W_out):
                    dw[ff, ...] += dout[nn, ff, jj, ii] * padded_x[nn,:,jj*stride:jj*stride+HH,ii*stride:ii*stride+WW]
                    
    # 其实就是按照之前的 这一层的输出等于w*x, 然后就是一个求导啦 对于dw的求导就是乘(点积)一下x；对于dx的求导就是点积一下w

    # Calculatde dx.
    # 计算dx的意义就是给后面的继续链式用
    # By chain rule dx is dout*w. We need to make dx same shape as padded x for the gradient calculation.
    for nn in range(N):
        for ff in range(F):
            for jj in range(H_out):
                for ii in range(W_out):
                    dx_temp[nn, :, jj*stride:jj*stride+HH,ii*stride:ii*stride+WW] += dout[nn, ff, jj,ii] * w[ff, ...]

        # 需要移除padding出的项
    # Remove the padding from dx so it matches the shape of x.
    dx = dx_temp[:, :, pad:H+pad, pad:W+pad]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    #print("?????")
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height = pool_param.get('pool_height')
    pool_width = pool_param.get('pool_width')
    stride = pool_param.get('stride')
    #print(N, C, H, W)
    #print("------------------")
    out_H = np.int(((H - pool_height) / stride) + 1)
    out_W = np.int(((W - pool_width) / stride) +1)
    
    out = np.zeros([N, C, out_H, out_W])
    
    for n in range(N):
        for c in range(C):
            for j in range(out_H):
                for i in range(out_W):
                    out[n, c, j, i] = np.max(x[n,c,j*stride:j*stride+pool_height, i*stride:i*stride+pool_width])
                    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    
    然后就是反向传播啦，这个其实就是找到最大值的位置，然后附上后面层的值就好啦

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    
    x, pool_param = cache
    N, C, H, W = x.shape
    extra_N, extra_C, dout_H, dout_W = dout.shape
    pool_height = pool_param.get('pool_height')
    pool_width = pool_param.get('pool_width')
    stride = pool_param.get('stride')
    
    dx = np.zeros(x.shape) #np.zeros_like(x)
    
    for n in range(N): #for each image
        for c in range(C): #for each channel
            for j in range(dout_H): #for each row of dout
                for i in range(dout_W):
                    
                    max_index = np.argmax(x[n, c, j*stride:j*stride+pool_height, i*stride:i*stride+pool_width])
                    
                    max_coord = np.unravel_index(max_index, [pool_height, pool_width])
                    
                    dx[n, c, j*stride:j*stride+pool_height, i*stride:i*stride+pool_width] [max_coord] = dout[n,c,j,i]
        
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.
    这个立体的batch normliaztion是什么鬼啊
    (20181214update：就是一个把四维的转化成二维的，以便可以用batch normlization（4D->2D)
    需要进行一个小小的转化，将(N,C,H,W)->(N*H*W, C)

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

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    temp_output, cache = batchnorm_forward(x.transpose(0, 3, 2, 1).reshape((N*H*W, C)), gamma, beta, bn_param)
    out = temp_output.reshape(N, W, H, C).transpose(0, 3, 2, 1)
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

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    dx_temp, dgamma, dbeta = batchnorm_backward_alt(dout.transpose(0, 3, 2, 1).reshape((N*H*W, C)), cache)
    dx = dx_temp.reshape(N, W, H, C).transpose(0, 3, 2, 1)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    
    计算立体的group normlization。
    
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    #########################################################################
    
    N, C, H, W = x.shape
    
    x_group = np.reshape(x, (N, G, C//G, H, W))
    
    mean = np.mean(x_group, axis = (2,3,4), keepdims = True)
    var = np.var(x_group, axis = (2,3,4), keepdims = True)
    x_groupnorm = (x_group-mean)/np.sqrt(var+eps)
    x_norm = np.reshape(x_groupnorm, (N,C,H,W))
    out = x_norm*gamma+beta
    cache = (G, x, x_norm, mean, var, beta, gamma, eps)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    
    
    
    G, x, x_norm, mean, var, beta, gamma, eps = cache
    
    N, C, H, W = x.shape
    
    dbeta = np.sum(dout, axis = (0, 2, 3), keepdims = True)
    dgamma = np.sum(dout*x_norm, axis = (0,2,3), keepdims = True)
    dx_norm = dout*gamma
    dx_groupnorm = dx_norm.reshape((N, G, C//G, H, W))
    # dvar
    x_group = x.reshape((N, G, C // G, H, W))
    dvar = np.sum(dx_groupnorm * -1 / 2 * (x_group - mean) / (var+eps)**(3/2), axis=(2,3,4), keepdims=True)
    
    N_GROUP = C//G*H*W
    dmean1 = np.sum(dx_groupnorm * -1 / np.sqrt(var + eps), axis = (2,3,4), keepdims = True)
    dmean2_var = dvar * -2 / N_GROUP * np.sum(x_group-mean, axis = (2,3,4), keepdims = True)
    dmean = dmean1 + dmean2_var
    
    dx_group1 = dx_groupnorm * 1 / np.sqrt(var + eps)
    dx_group2_mean = dmean * 1 / N_GROUP
    dx_group3_var = dvar * 2.0 / N_GROUP * (x_group - mean)
    dx_group = dx_group1 + dx_group2_mean + dx_group3_var
    
                                            
    dx = dx_group.reshape((N,C,H,W))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.
    #计算SVM分类器的损失函数

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0 #第y[i]这一行原本是肯定大于0的（f(y[i]-f[y[i]+delta])），但这里我们显然是不需要这个的
    loss = np.sum(margins) / N
    dx = np.zeros_like(x)
    dx[margins > 0] = 1 #自己本身的权重去减掉1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
