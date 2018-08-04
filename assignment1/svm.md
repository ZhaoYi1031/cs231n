## Assignment1-SVM线性分类器


主要内容：

1. 完成一个使用向量方法计算svm损失函数; 
2. 完成一个使用向量方法来分析梯度;
3. 使用数学方法来检查梯度
4. 用验证集来微调学习率和正则项;
5. 使用随机梯度下降法来优化损失函数; 6. 可视化最后学习到的权重

代码地址：

http://35.227.144.240:8888/tree/cs231n/assignment1
https://github.com/ZhaoYi1031/cs231n/tree/master/assignment1


*首先应该说的是这个SVM只是用到了SVM损失函数，不是支持向量机的那个SVM流程。*

SVM分类器的损失函数的计算公式如下：

$$L = \frac{1}{N}\sum_{i}\sum_{j\neq y_i}[max(0,  f_j(x_i; W)-f_{y_i}(x_i;W)+\triangle)] + \lambda \sum_k \sum_lW_{k,l}^2$$

这个定义可以看课程的视频，简单而言，就是如果W*X计算后的得分+1大于正确标签(也就是yi)的得分（这个就相当于预测错了，即怎么可能比真实的标签值的得分还要大），那么就加上这个偏差，否则无视（我们可以这么理解，对于小的显然是应该的，所以就不应该加到损失函数里，否则就是加上偏差）

梯度的数值计算(numeric computing)方法：

$$ \frac{df(x)}{dx} = \lim\limits_{h\to0} \frac{f(x+h)-f(x)}{h}$$

梯度的微分分析计算：

$$\nabla{L_i} = -(\sum_{j\neq{y_i}}1(w_j^Tx_i-w_{y_i}^T + \triangle > 0)) x_i$$

其中1(x)是示性函数，当x为真的时候函数值为1，否则为0.

在SVM分类器中，我们这里处理训练集、验证集、测试集之外又从训练集中随机选择500个样本作为development set，在最终的训练和预测之前，我们都使用这个小的数据集，当然，直接使用完整的训练集也是可以的，不过就是花费的时间有点多。（验证集在我们的代码中的作用就是在不同的学习率和正则化参数的结合里找到最好的组合，然后去检验测试集）经过将数据拉平后的数据的shape如下：

```
training data shape : (49000, 3072)
validation data shape : (1000, 3072)
test data shape : (1000, 3072)
development data shape : (500, 3072)
```

在线性分类中，我们使用的是随机梯度下降(stochastic gradient desecnt) 随机梯度下降是与批量梯度下降区分开的两种梯度下降方式。批量梯度下降是指每一步去按照参数例如是theta的梯度负方向去更新，我们用到的是全局的最优解，即每一次更新都要用到所有训练集的数据，所以它的迭代速度是很慢的。而随机梯度下降是选取其中的小部分样本去进行计算梯度，这样虽然迭代的不是全局最优方向，但是大的整体的方向是全局最优解的，最终的结果往往也在全局最优解附近。在我们的例子中，训练集本来的大小是49000\*3073的，如果要更新参数矩阵（10\*3073)的大小，计算loss和grad是非常耗时的，所以我们随机选取200个训练集的例子，构成200*3073的batch去做随机梯度下降，时间会提快很多。我经过实验，batch_size选取200的时候需要不到4秒，而设置成1000的时候需要15.56秒。

下面是代码部分：

## svm.ipvnb

*Complete and hand in this completed worksheet (including its outputs and any supporting code outside of the worksheet) with your assignment submission. For more details see the [assignments page](http://vision.stanford.edu/teaching/cs231n/assignments.html) on the course website.*

In this exercise you will:
    
- implement a fully-vectorized **loss function** for the SVM
- implement the fully-vectorized expression for its **analytic gradient**
- **check your implementation** using numerical gradient
- use a validation set to **tune the learning rate and regularization** strength
- **optimize** the loss function with **SGD**
- **visualize** the final learned weights



```python
# Run some setup code for this notebook.

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

from __future__ import print_function

# This is a bit of magic to make matplotlib figures appear inline in the
# notebook rather than in a new window.
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


## CIFAR-10 Data Loading and Preprocessing


```python
# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
```

    Training data shape:  (50000, 32, 32, 3)
    Training labels shape:  (50000,)
    Test data shape:  (10000, 32, 32, 3)
    Test labels shape:  (10000,)



```python
# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
```


![png](https://i.loli.net/2018/08/04/5b6584fd79e60.png)





```python
# Split the data into train, val, and test sets. In addition we will
# create a small development set as a subset of the training data;
# we can use this for development so our code runs faster.
# 把我们的数据集分成训练集、验证集和测试集。除此之外，我们也创建了一个小的发展集合(?)作为训练集的子集，我们使用这个作为法阵因此我们的代码跑的更快
num_training = 49000 #训练集的个数
num_validation = 1000 #验证集的个数
num_test = 1000 #测试集的个数
num_dev = 500 #development集合的个数

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation) #选取数据集[49000, 50000]区间内的数据作为我们的验证集
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask] #[0,49000]作为我们的训练集
y_train = y_train[mask]

# We will also make a development set, which is a small subset of
# the training set.
mask = np.random.choice(num_training, num_dev, replace=False) #从之前已经选好的训练集中随机抽出500个作为我们的dev集
#print(mask)
X_dev = X_train[mask]
y_dev = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask] #选取原来测试集（大小是10000）中的前1000作为我们的测试集
y_test = y_test[mask]

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
```

    Train data shape:  (49000, 32, 32, 3)
    Train labels shape:  (49000,)
    Validation data shape:  (1000, 32, 32, 3)
    Validation labels shape:  (1000,)
    Test data shape:  (1000, 32, 32, 3)
    Test labels shape:  (1000,)



```python
# Preprocessing: reshape the image data into rows
# 数据预处理：把数据压缩成一行
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# As a sanity check, print out the shapes of the data
print('Training data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)
print('dev data shape: ', X_dev.shape)
```

    Training data shape:  (49000, 3072)
    Validation data shape:  (1000, 3072)
    Test data shape:  (1000, 3072)
    dev data shape:  (500, 3072)



```python
# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
# 初始： 计算平均的图像（OS:这TM的平均图像是什么鬼）
mean_image = np.mean(X_train, axis=0)
print(mean_image[:10]) # print a few of the elements
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image
plt.show()
```

    [130.64189796 135.98173469 132.47391837 130.05569388 135.34804082
     131.75402041 130.96055102 136.14328571 132.47636735 131.48467347]



![png](https://i.loli.net/2018/08/04/5b6584fd46922.png)


```python
# second: subtract the mean image from train and test data
# 把训练集和测试集的数据分别减去平均图片的数值
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image
```


```python
# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
# 给每一张图片增加截距项(bias trick)，从而从3072维变成了3073维。因此我们的SVM只需要关注优化一个单独的权重矩阵W
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)
```

    (49000, 3073) (1000, 3073) (1000, 3073) (500, 3073)


## SVM Classifier

Your code for this section will all be written inside **cs231n/classifiers/linear_svm.py**. 

As you can see, we have prefilled the function `compute_loss_naive` which uses for loops to evaluate the multiclass SVM loss function. 


```python
# Evaluate the naive implementation of the loss we provided for you:
from cs231n.classifiers.linear_svm import svm_loss_naive
import time

# generate a random SVM weight matrix of small numbers
W = np.random.randn(3073, 10) * 0.0001 

loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005) #grad: 3073*10
print('loss: %f' % (loss, ))
```

    loss: 9.019190


The `grad` returned from the function above is right now all zero. Derive and implement the gradient for the SVM cost function and implement it inline inside the function `svm_loss_naive`. You will find it helpful to interleave your new code inside the existing function.

To check that you have correctly implemented the gradient correctly, you can numerically estimate the gradient of the loss function and compare the numeric estimate to the gradient that you computed. We have provided code that does this for you:


```python
# Once you've implemented the gradient, recompute it with the code below
# and gradient check it with the function we provided for you

# Compute the loss and its gradient at W.
loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)

# Numerically compute the gradient along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should match
# almost exactly along all dimensions.
from cs231n.gradient_check import grad_check_sparse
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad)

# do the gradient check once again with regularization turned on
# you didn't forget the regularization gradient did you?
loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad)
```

    numerical: -6.939025 analytic: -6.939025, relative error: 7.725614e-11
    numerical: 7.517536 analytic: 7.517536, relative error: 3.421724e-11
    numerical: 12.753084 analytic: 12.753084, relative error: 7.566703e-12
    numerical: -9.788667 analytic: -9.788667, relative error: 1.958777e-11
    numerical: -0.740939 analytic: -0.740939, relative error: 3.379269e-10
    numerical: -6.175633 analytic: -6.175633, relative error: 1.279184e-11
    numerical: 12.691954 analytic: 12.691954, relative error: 2.927269e-11
    numerical: -5.732234 analytic: -5.732234, relative error: 2.624024e-11
    numerical: -12.942757 analytic: -12.942757, relative error: 9.279487e-12
    numerical: 25.709632 analytic: 25.709632, relative error: 2.208442e-11
    numerical: 21.557588 analytic: 21.557588, relative error: 1.900156e-13
    numerical: 0.441689 analytic: 0.441689, relative error: 3.188058e-10
    numerical: -11.126465 analytic: -11.126465, relative error: 1.136767e-11
    numerical: 21.561594 analytic: 21.561594, relative error: 9.181010e-12
    numerical: 0.952637 analytic: 0.952637, relative error: 6.582794e-11
    numerical: -1.761509 analytic: -1.761509, relative error: 2.131017e-10
    numerical: 2.558565 analytic: 2.558565, relative error: 2.596249e-10
    numerical: 0.701399 analytic: 0.701399, relative error: 4.673075e-10
    numerical: -10.093727 analytic: -10.093727, relative error: 4.816963e-11
    numerical: 46.885397 analytic: 46.885397, relative error: 4.692407e-12


### Inline Question 1:
It is possible that once in a while a dimension in the gradcheck will not match exactly. What could such a discrepancy be caused by? Is it a reason for concern? What is a simple example in one dimension where a gradient check could fail? How would change the margin affect of the frequency of this happening? *Hint: the SVM loss function is not strictly speaking differentiable*

**Your Answer:** *fill this in.*


```python
# Next implement the function svm_loss_vectorized; for now only compute the loss;
# we will implement the gradient in a moment.
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))

from cs231n.classifiers.linear_svm import svm_loss_vectorized
tic = time.time()
loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# The losses should match but your vectorized implementation should be much faster.
print('difference: %f' % (loss_naive - loss_vectorized))
```

    Naive loss: 9.019190e+00 computed in 0.084337s
    Vectorized loss: 9.019190e+00 computed in 0.003015s
    difference: -0.000000



```python
# Complete the implementation of svm_loss_vectorized, and compute the gradient
# of the loss function in a vectorized way.

# The naive implementation and the vectorized implementation should match, but
# the vectorized version should still be much faster.
tic = time.time()
_, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Naive loss and gradient: computed in %fs' % (toc - tic))

tic = time.time()
_, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Vectorized loss and gradient: computed in %fs' % (toc - tic))

# The loss is a single number, so it is easy to compare the values computed
# by the two implementations. The gradient on the other hand is a matrix, so
# we use the Frobenius norm to compare them.
difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('difference: %f' % difference)
```

    Naive loss and gradient: computed in 0.085203s
    Vectorized loss and gradient: computed in 0.002733s
    difference: 0.000000


### Stochastic Gradient Descent

We now have vectorized and efficient expressions for the loss, the gradient and our gradient matches the numerical gradient. We are therefore ready to do SGD to minimize the loss.


```python
# In the file linear_classifier.py, implement SGD in the function
# LinearClassifier.train() and then run it with the code below.
from cs231n.classifiers import LinearSVM
svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                      num_iters=1500, verbose=True)
toc = time.time()
print('That took %fs' % (toc - tic))
```

    iteration 0 / 1500: loss 776.839683
    iteration 100 / 1500: loss 465.207123
    iteration 200 / 1500: loss 282.523762
    iteration 300 / 1500: loss 173.023402
    iteration 400 / 1500: loss 105.895383
    iteration 500 / 1500: loss 65.952037
    iteration 600 / 1500: loss 42.363984
    iteration 700 / 1500: loss 27.530621
    iteration 800 / 1500: loss 18.056885
    iteration 900 / 1500: loss 13.621412
    iteration 1000 / 1500: loss 10.253616
    iteration 1100 / 1500: loss 8.287845
    iteration 1200 / 1500: loss 7.222646
    iteration 1300 / 1500: loss 6.335761
    iteration 1400 / 1500: loss 6.264345
    That took 3.957306s



```python
# A useful debugging strategy is to plot the loss as a function of
# iteration number:
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()
```


```python
# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
y_train_pred = svm.predict(X_train)
print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
y_val_pred = svm.predict(X_val)
print('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))
```

    training accuracy: 0.384143
    validation accuracy: 0.394000



```python
# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of about 0.4 on the validation set.
learning_rates = [1e-7, 5e-5]
regularization_strengths = [2.5e4, 5e4]

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.

################################################################################
# TODO:                                                                        #
# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a linear SVM on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the LinearSVM object that achieves this  #
# accuracy in best_svm.                                                        #
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the SVMs don't take much time to train; once you are #
# confident that your validation code works, you should rerun the validation   #
# code with a larger value for num_iters.                                      #
################################################################################
from cs231n.classifiers import LinearSVM

for i in learning_rates:
    for j in regularization_strengths:
        svm = LinearSVM()
        loss_hist = svm.train(X_train, y_train, learning_rate=i, reg=j,
                      num_iters=1000, verbose=True) #这个输出是一个一维数组，长度是num_iter，即每次迭代的loss大小
        y_train_pred = svm.predict(X_train) 
        train_accuracy = np.mean(y_train == y_train_pred) #计算出训练集的准确度
        print('training accuracy: %f' % train_accuracy)
        y_val_pred = svm.predict(X_val)
        validation_accuracy = np.mean(y_val == y_val_pred) #计算出验证集的准确度
        print('validation accuracy: %f' % validation_accuracy)
        results[(i, j)] = (train_accuracy, validation_accuracy) #把结果保存到map里面
        if validation_accuracy > best_val:
            best_val = validation_accuracy #max(best_val, validation_accuracy)
            best_svm = svm
        
print(results)       

################################################################################
#                              END OF YOUR CODE                                #
################################################################################
    
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val)
```

    iteration 0 / 1000: loss 788.138863
    iteration 100 / 1000: loss 471.064637
    iteration 200 / 1000: loss 286.701304
    iteration 300 / 1000: loss 174.451062
    iteration 400 / 1000: loss 107.007136
    iteration 500 / 1000: loss 66.252252
    iteration 600 / 1000: loss 42.181601
    iteration 700 / 1000: loss 27.369910
    iteration 800 / 1000: loss 18.549549
    iteration 900 / 1000: loss 13.325149
    training accuracy: 0.371388
    validation accuracy: 0.375000
    iteration 0 / 1000: loss 1545.334268
    iteration 100 / 1000: loss 563.662067
    iteration 200 / 1000: loss 208.210732
    iteration 300 / 1000: loss 79.989896
    iteration 400 / 1000: loss 32.551170
    iteration 500 / 1000: loss 15.461266
    iteration 600 / 1000: loss 9.297513
    iteration 700 / 1000: loss 7.099414
    iteration 800 / 1000: loss 6.608425
    iteration 900 / 1000: loss 6.535689
    training accuracy: 0.366388
    validation accuracy: 0.382000
    iteration 0 / 1000: loss 786.860024
    iteration 100 / 1000: loss 1396.916401
    iteration 200 / 1000: loss 1566.884792
    iteration 300 / 1000: loss 1089.567531
    iteration 400 / 1000: loss 1279.183220
    iteration 500 / 1000: loss 1194.818357
    iteration 600 / 1000: loss 1495.081416
    iteration 700 / 1000: loss 1204.803013
    iteration 800 / 1000: loss 1739.252798
    iteration 900 / 1000: loss 1727.826417
    training accuracy: 0.149224
    validation accuracy: 0.159000
    iteration 0 / 1000: loss 1553.560336
    iteration 100 / 1000: loss 767179154186042438517259400694116384768.000000
    iteration 200 / 1000: loss 126808569909578629558869403150867496252670460369197847174562238270254612480.000000
    iteration 300 / 1000: loss 20960441006212435335625270607727047335944639666135455198468707870095918010718834856481057228430819427463725056.000000
    iteration 400 / 1000: loss 3464593027807070604448694724055332493080626555558948149861133776986950379847340559297777900903092088485059915563010719539151652999411562142760960.000000
    iteration 500 / 1000: loss 572669479844040140531150965412253247854281638243698610762182146875580487294483893983025271519399128080300457451093106706311442636136237626241162722981273022549497517934878398611456.000000
    iteration 600 / 1000: loss 94657678553495538100790047876475721064548133264618481454588930372193141804931246640908906758575703767951800038244687678546176731478748136543158787086471205318654514796313723527219096477231545658740205225870714994688.000000
    iteration 700 / 1000: loss 15646156159006525534845259654378820236619672094531013391707910376948248836447242159692045508397485685621313900880592613583644398132339412812677634390295198702372183572066747421091648830289362484254581723194192806723619492017712503007896189704832286720.000000
    iteration 800 / 1000: loss 2586184304252384843770570436555716296801850738793214579927852875152111553136732572731360684203355230075315057046279501624337591928280017387272568090702807731855420362224167896064959050919412414491417977986048419561363733481606447026132837195787339172457353415429613613876883556849942528.000000


    /home/Mr.ZY/github/cs231n/assignment1/cs231n/classifiers/linear_svm.py:94: RuntimeWarning: overflow encountered in double_scalars
      #print(scores.shape)
    /home/Mr.ZY/anaconda3/lib/python3.5/site-packages/numpy/core/_methods.py:32: RuntimeWarning: overflow encountered in reduce
      return umr_sum(a, axis, dtype, out, keepdims)
    /home/Mr.ZY/github/cs231n/assignment1/cs231n/classifiers/linear_svm.py:94: RuntimeWarning: overflow encountered in multiply
      #print(scores.shape)


    iteration 900 / 1000: loss inf
    training accuracy: 0.065265
    validation accuracy: 0.064000
    {(1e-07, 25000.0): (0.3713877551020408, 0.375), (5e-05, 50000.0): (0.06526530612244898, 0.064), (1e-07, 50000.0): (0.3663877551020408, 0.382), (5e-05, 25000.0): (0.14922448979591837, 0.159)}
    lr 1.000000e-07 reg 2.500000e+04 train accuracy: 0.371388 val accuracy: 0.375000
    lr 1.000000e-07 reg 5.000000e+04 train accuracy: 0.366388 val accuracy: 0.382000
    lr 5.000000e-05 reg 2.500000e+04 train accuracy: 0.149224 val accuracy: 0.159000
    lr 5.000000e-05 reg 5.000000e+04 train accuracy: 0.065265 val accuracy: 0.064000
    best validation accuracy achieved during cross-validation: 0.382000



```python
# # Visualize the cross-validation results
# import math


# x_scatter = [math.log10(x[0]) for x in results]
# y_scatter = [math.log10(x[1]) for x in results]

# # plot training accuracy
# marker_size = 100
# colors = [results[x][0] for x in results]
# plt.subplot(2, 1, 1)
# plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
# plt.colorbar()
# plt.xlabel('log learning rate')
# plt.ylabel('log regularization strength')
# plt.title('CIFAR-10 training accuracy')

# # plot validation accuracy
# colors = [results[x][1] for x in results] # default size of markers is 20
# plt.subplot(2, 1, 2)
# plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
# plt.colorbar()
# plt.xlabel('log learning rate')
# plt.ylabel('log regularization strength')
# plt.title('CIFAR-10 validation accuracy')
# plt.show()
```


```python
import math
x_scatter=[math.log10(x[0]) for x in results] #1
y_scatter=[math.log10(x[1]) for x in results] #2
sz=[results[x][0]*1500 for x in results]  #3
plt.subplot(1,2,1)
plt.scatter(x_scatter,y_scatter,sz)
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('cifar10 training accuracy')
sz=[results[x][1]*1500 for x in results]
plt.subplot(1,2,2)
plt.scatter(x_scatter,y_scatter,sz)
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('cifar10 validation accuracy')
plt.show()
# 面积的大小代表准确率，可以看到不同的学习率和正则化
```


```python
# Evaluate the best svm on test set
y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)
```


```python
# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength, these may
# or may not be nice to look at.
w = best_svm.W[:-1,:] # strip out the bias
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)
      
    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])
```

### Inline question 2:
Describe what your visualized SVM weights look like, and offer a brief explanation for why they look they way that they do.

**Your answer:** *fill this in*

## LinearClassifier.py(SGD部分）

```
class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = 0.001 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
      X_batch = None
      y_batch = None
      
      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      #print(dim*batch_size) #print(X.flatten().shape)
      #X_batch = np.reshape(np.random.choice(X.flatten(), int(batch_size*dim), replace=True), (-1, dim)) #print(X_batch)
      #y_batch = np.random.choice(y, batch_size, replace=True) #print(y_batch)
      sample_index = np.random.choice(num_train, batch_size, replace=True) #从[0,num_train]中选取batch_size个数，replace为True代表里面的数可以重复
      X_batch = X[sample_index, :]
      y_batch = y[sample_index]
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      #print(grad)
      loss_history.append(loss)

      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      self.W -= learning_rate*grad
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history
```

## Linear_svm.py

两种计算梯度的方法，naive版本的是没有向量化的。

```
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
```


