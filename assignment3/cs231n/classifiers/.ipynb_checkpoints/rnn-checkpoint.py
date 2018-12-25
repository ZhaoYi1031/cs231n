from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class CaptioningRNN(object):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.
    
    一个captionRNN产生图片的描述使用RNN

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.
    
    RNN接受一个输入大小是D，有V个单词，作用在长度为T，有RNN隐藏层的大小是H，使用词向量的维度是W，并且在一个minibatch为N的大小上计算

    Note that we don't use any regularization for the CaptioningRNN.
    
    注意到我们不使用任何正则项。
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        """
        Construct a new CaptioningRNN instance.
        
        构造一个新的CpationRNN的实例。

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
          
          一个dict表示了这个词典
          它包含了V个入口，并且映射每个string到一个独特的整数在范围[0,v]
        - input_dim: Dimension D of input image feature vectors.
        
            输入维度：D维的图像的vector
        - wordvec_dim: Dimension W of word vectors.
            W维的词向量的
        
        - hidden_dim: Dimension H for the hidden state of the RNN.
            H维的隐藏层的信息
        
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        
            RNN的类型，rnn或者lstm
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.
        
        计算RNN的训练的损失。我们输入图像特征和事实的描述。使用一个RNN来计算loss和梯度

        Inputs:
        - features: Input image features, of shape (N, D) #特征，输入的图像的特征，形状是(N,D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V
        
        - 描述：实际的描述，一个整数array，形状是（N，T）每一个元素是范围(0,T)

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        
        # 把描述cut成两个部分：captions_in有除了最后一个单词的所有并且将会是RNN的输入；captions_out除了第一个单词的所有并且是我们期望RNN去生成的。这些是一个
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (1) 使用一个线性转移层来计算初始的隐藏层的状态。将产生一个形状是(N,H)的array
        # (2) Use a word embedding layer to transform the words in captions_in     #
        # (2) 使用一个word embedding层来换换在captions_in中的单词从下标到vector?
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        
        # (3)使用或者是一个vanilla RNN或者LSTM（取决于self.cell_type)来处理输入单词vector的序列并且产生所有时间戳的隐藏层状态。产生一个array，形状是（N，T，H）
        
        
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).       
        
        # (4) 使用一个临时的线性层来计算词典上的scores在每一个时间戳使用隐藏层状态，
        
        #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        
        # （5） 使用暂时的softmax层来计算loss使用captions_out, 忽略output单词是NULL的这种
            
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        #                                                                          #
        # Note also that you are allowed to make use of functions from layers.py   #
        # in your implementation, if needed.                                       #
        ############################################################################
        
        #print("captions_in =", captions_in)
        
        #(1) 使用一个线性转移层来计算初始的隐藏层的状态。将产生一个形状是(N,H)的array
        
        h0, cache_h0 = affine_forward(features, W_proj, b_proj)
        
        #
        
        #(2) Use a word embedding layer to transform the words in captions_in from indices to vectors, giving an array of shape (N, T, W).  
        embedded_captions, cache_embedded_captions = word_embedding_forward(captions_in, W_embed)
        
        #print("embedded_captions.shape = ", embedded_captions.shape)
        #print("embedded_captions =", embedded_captions)
        
        #(3)使用或者是一个vanilla RNN或者LSTM（取决于self.cell_type)来处理输入单词vector的序列并且产生所有时间戳的隐藏层状态。产生一个array，形状是（N，T，H）
        if self.cell_type == 'rnn':
            rnn_output, rnn_cache = rnn_forward(embedded_captions, h0, Wx, Wh, b)
        else:
            lstm_output, lstm_cache = lstm_forward(embedded_captions, h0, Wx, Wh, b)
            
        #print("rnn_output = ", rnn_output)
        
        #(4) 使用一个临时的线性层来计算词典上的scores在每一个时间戳使用隐藏层状态，
        
        scores, cache_scores = temporal_affine_forward(rnn_output, W_vocab, b_vocab)
        
        #(5) 使用暂时的softmax层来计算loss使用captions_out, 忽略output单词是NULL的这种
        loss, dsoftmax = temporal_softmax_loss(scores, captions_out, mask, False)
        
        
        #-------------------------下面是反向传播了-------------------------#
        dx_score, dW_vocab, db_vocab = temporal_affine_backward(dsoftmax, cache_scores)
        grads['W_vocab'], grads['b_vocab'] = dW_vocab, db_vocab
        
        if self.cell_type == 'rnn':
            dx_cell, dh0, dWx_rnn, dWh_rnn, db_rnn = rnn_backward(dx_score, rnn_cache)
            grads['Wx'], grads['Wh'], grads['b'] = dWx_rnn, dWh_rnn, db_rnn
        else:
            dx_cell, dh0, dWx_lstm, dWh_lstm, db_lstm = lstm_backward(dx_score, rnn_cache)
            grads['Wx'], grads['Wh'], grads['b'] = dWx_lstm, dWh_lstm, db_lstm
        
        dW_embedded = word_embedding_backward(dx_cell, cache_embedded_captions)
        grads['W_embed'] = dW_embedded
        
        dx, dW_proj, db_proj = affine_backward(dh0, cache_h0)
        grads['W_proj'], grads['b_proj'] = dW_proj, db_proj
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.
        
        执行一个test-time的前向传播给这个模型，sampling摘要作为输入给输入的特征向量。
        
        在每一个时间戳，我们embed当前词语，传递它和之前的状态给RNN去获取下一个隐藏层的模型，使用隐藏层去得到所有词典单词的得分。
        并且选择得分最高的单词作为下一个单词。初始的隐藏层的状态用一个线性转化，最开始的token是<START>

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.
        
        对于LSTM你还需要跟踪cell状态。

        Inputs:
        - features: Array of input image features of shape (N, D). 输入特征
        - max_length: Maximum length T of generated captions. 生成的caption的最大长度
 
        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element #每一个元素在范围[0,V]
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        print("N=", N)
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        print("captions.shape", captions.shape)
        
        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     (the word index) to the appropriate slot in the captions variable   #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        #                                                                         #
        # NOTE: we are still working over minibatches in this function. Also if   #
        # you are using an LSTM, initialize the first cell state to zeros.        #
        ###########################################################################
        
        current_h, cache_current_h = affine_forward(features, W_proj, b_proj)
        
        current_c = np.zeros_like(current_h)
        
        words = np.zeros(N, dtype = int)
        
        words.fill(self._start)
        
        for step in range(max_length):
            #(1) Embed the previous word using the learned word embeddings
            embedded_words, _ = word_embedding_forward(words, W_embed)
            
            #(2) Make an RNN step using the previous hidden state and the embedded   #
            #current word to get the next hidden state.      
            if self.cell_type=='rnn':
                current_h, _ = rnn_step_forward(embedded_words, current_h, Wx, Wh, b)
                
            scores = current_h.dot(W_vocab) + b_vocab
            
        #    Select the word with the highest score as the next word, writing it #
        #     (the word index) to the appropriate slot in the captions variable 
        
            captions[:, step] = np.argmax(scores, axis = 1) #用得分最大的那个作为下一个隐藏层的输入
            
            print("captions[:, step]=", captions[:, step])
            words = captions[:, step] 
            print("step = ",step)
            print(words)
            
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions
