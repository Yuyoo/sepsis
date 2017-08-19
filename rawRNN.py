# -*- coding:utf-8 -*-
import cPickle as pickle
import random
import sys
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from sklearn.metrics import roc_auc_score
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def unzip(zipped):
    new_params = OrderedDict()
    for key, value in zipped.iteritems():
        new_params[key] = value.get_value()
    return new_params

'''
张量元素类型转换
将张量元素转换成config中设置的float类型，以便在显存中进行运算
'''
def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def get_random_weight(dim1, dim2, left=-0.1, right=0.1):
    return np.random.uniform(left, right, (dim1, dim2)).astype(config.floatX)

'''
初始化网络连接参数
params
    options：网络超参数
return
    网络连接参数的字典params
'''
def init_params(options):
    params = OrderedDict()  # 实例化有序字典params，字典顺序按插入顺序排序

    inputDimSize = options['inputDimSize']
    hiddenDimSize = options['hiddenDimSize']  # hidden layer does not need an extra space

    # 加载embfile文件并将里面数组元素转换为config中规定的浮点类型
    params['W_emb'] = np.array(pickle.load(open(options['embFile'], 'rb'))).astype(config.floatX)

    # W,U,b是RNN层的连接参数
    # 这里除考虑embed的原始输入维度外，还要加入时间间隔信息，因此行数+1
    # 这里讲r、z、h三组激活函数的连接参数全部放在一起，方便后面一起运算，因此连接矩阵列数为3 * hiddenDimSize
    params['W_gru'] = get_random_weight(embDimSize + 1, 3 * hiddenDimSize)
    params['U_gru'] = get_random_weight(hiddenDimSize, 3 * hiddenDimSize)
    params['b_gru'] = np.zeros(3 * hiddenDimSize).astype(config.floatX)

    # W,b是全连接层或softmax层连接参数
    params['W_logistic'] = get_random_weight(hiddenDimSize, 1)
    params['b_logistic'] = np.zeros((1,), dtype=config.floatX)

    return params

'''
将网络连接参数转化为符号变量
shared符号变量可以存储在显存中
shared指向显存中的一块区域，这块区域在运算中是共享的，所以常常在运算中用来存储权值参数
'''
def init_tparams(params):
    tparams = OrderedDict()
    for key, value in params.iteritems():
        if key == 'W_emb': continue
        tparams[key] = theano.shared(value, name=key)
    return tparams

'''
dropout层
    :param
        state_before：网络输出特征
        use_noise：噪声控制矩阵
        trng：随机器对象
    :return
        proj：dropout后的特征
'''
def dropout_layer(state_before, use_noise, trng):
    proj = T.switch(use_noise, (state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype)),
                    state_before * 0.5)
    return proj

'''
神经元连接参数切片函数
原[W,U,b]是r,z,h三组合并的连接矩阵
将合并的r，z，h激活函数的线性部分划分为三组

参数
    _x:激活函数的线性部分，即_x = W*X_emb + b.
    n：划分索引
    dim：隐含层数量，即单个连接矩阵的维度    
'''
def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]

'''
gru神经层构造函数
参数
    tparms：网络参数权值与偏置等
    emb：网络RNN层输入张量，张量由外往内的维度分别是单个序列时间步，序列数，每个时间步每个人的embedded输入
    option：常规时不变超参数
    mask：掩膜向量，用来控制序列的长度
'''
def gru_layer(tparams, emb, options, mask=None):
    hiddenDimSize = options['hiddenDimSize']
    timesteps = emb.shape[0]    # 取出输入的时间序列步数
    if emb.ndim == 3:           # 判断embedded输入的维度确定样本数
        n_samples = emb.shape[1]    # 取出输入样本数
    else:
        n_samples = 1

    '''
    序列迭代函数
    参数
        stepMask：时间步掩膜
        wx：输入门激活函数线性部分
        h：神经元状态
        U_gru：状态门连接系数
    :return
        下一个序列的状态
    '''
    def stepFn(stepMask, wx, h, U_gru):
        uh = T.dot(h, U_gru)
        r = T.nnet.sigmoid(_slice(wx, 0, hiddenDimSize) + _slice(uh, 0, hiddenDimSize))
        z = T.nnet.sigmoid(_slice(wx, 1, hiddenDimSize) + _slice(uh, 1, hiddenDimSize))
        h_tilde = T.tanh(_slice(wx, 2, hiddenDimSize) + r * _slice(uh, 2, hiddenDimSize))
        h_new = z * h + ((1. - z) * h_tilde)
        h_new = stepMask[:, None] * h_new + (1. - stepMask)[:, None] * h
        return h_new

    Wx = T.dot(emb, tparams['W_gru']) + tparams['b_gru']    # 求激活函数的线性部分

    '''
    遍历序列执行函数stepFn，其中sequences，outputs_info，non_sequences作为参数传入到fn中
    :param
        fn:fn是一个lambda或者def函数，描述了一步scan操作的运算式，
            运算式的输入参数按照sequences, outputs_info, non_sequences的顺序，
            运算式的输出作为theano.scan的返回值。
        sequences:sequences是一个theano variables或者dictionaries的列表。
            字典对象的结构为{‘variable’：taps}，其中taps是一个整数列表。
            ’sequences’列表中的所有Theano variable会被自动封装成一个字典，此时taps被设置成[0]。
            比如sequences = [ dict(input= Sequence1, taps = [-3,2,-1]), Sequence2， dict(input = Sequence3, taps = 3) ]， 
            映射到scan输入参数为Sequence1[t-3]，Sequence1[t+2]，Sequence1[t-1]，Sequence2[t]，Sequence3[t+3]。
        non_sequences:non_sequences 是一个‘常量’参数列表，这里所谓的‘常量’是相对于‘outputs_info’中的参数更新而言的，
            代表了一步scan操作中不会被更新的变量。
        n_steps：n_steps参数是一个int或者theano scalar，代表了scan操作的迭代次数。
            如果n_steps参数未指定，scan会根据他的输入参数自动计算出迭代步数 
    '''
    results, updates = theano.scan(fn=stepFn, sequences=[mask, Wx],
                                   outputs_info=T.alloc(numpy_floatX(0.0), n_samples, hiddenDimSize),
                                   non_sequences=[tparams['U_gru']], name='gru_layer', n_steps=timesteps)

    return results[-1]  # 返回隐含层最后一个序列的状态

'''
模型构造函数
    :params
        tparams：网络连接参数
        options：网络超参数
        Wemb：网络输入
    :returns
        use_noise：噪声控制矩阵，用于dropout层，控制训练与测试过程的池化操作
        x：特征输入
        t：时间输入
        mask：序列掩膜，控制短序列的后续状态
        y：真实标签
        p_y_given_x：网络输出的条件概率
        cost：损失
'''
def build_model(tparams, options, Wemb):
    trng = RandomStreams(123)       # 随机矩阵实例化
    use_noise = theano.shared(numpy_floatX(0.))     # 定义shared参数

    x = T.matrix('x', dtype='int32')
    t = T.matrix('t', dtype=config.floatX)
    mask = T.matrix('mask', dtype=config.floatX)
    y = T.vector('y', dtype='int32')

    n_timesteps = x.shape[0]    # 获取时间步数
    n_samples = x.shape[1]      # 获取样本数

    # 重构网络输入
    x_emb = Wemb[x.flatten()].reshape([n_timesteps, n_samples, options['embDimSize']])
    x_t_emb = T.concatenate([t.reshape([n_timesteps, n_samples, 1]), x_emb],
                            axis=2)  # 在原输入中加入时间维度信息

    # 定义RNN层
    proj = gru_layer(tparams, x_t_emb, options, mask=mask)

    # dropout层
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    # 损失函数
    p_y_given_x = T.nnet.sigmoid(T.dot(proj, tparams['W_logistic']) + tparams['b_logistic'])    # 求网络输出
    L = -(y * T.flatten(T.log(p_y_given_x)) + (1 - y) * T.flatten(T.log(1 - p_y_given_x)))  # 损失函数
    cost = T.mean(L)    # 求所有样本损失的均值作为cost

    # L2正则化
    if options['L2_reg'] > 0.:
        cost += options['L2_reg'] * (tparams['W_logistic'] ** 2).sum()

    return use_noise, x, t, mask, y, p_y_given_x, cost


# 加载特征序列文件，标签文件，以及采样时间点文件（可选），并构造训练集，验证集以及测试集
def load_data(seqFile, labelFile, timeFile=''):
    sequences = np.array(pickle.load(open(seqFile, 'rb')))
    labels = np.array(pickle.load(open(labelFile, 'rb')))
    if len(timeFile) > 0:
        times = np.array(pickle.load(open(timeFile, 'rb')))

    dataSize = len(labels)
    ind = np.random.permutation(dataSize)   # 生成dataSize大小的随机向量
    nTest = int(0.10 * dataSize)
    nValid = int(0.10 * dataSize)

    test_indices = ind[:nTest]
    valid_indices = ind[nTest:nTest + nValid]
    train_indices = ind[nTest + nValid:]

    # 划分数据集
    train_set_x = sequences[train_indices]
    train_set_y = labels[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    valid_set_x = sequences[valid_indices]
    valid_set_y = labels[valid_indices]
    train_set_t = None
    test_set_t = None
    valid_set_t = None

    if len(timeFile) > 0:
        train_set_t = times[train_indices]
        test_set_t = times[test_indices]
        valid_set_t = times[valid_indices]

    # 将列表按列表内元素的长度进行排序，默认升序，返回的列表存储原来内层列表的索引
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    # 将数据集按时间序列采样长度进行重新排序
    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    if len(timeFile) > 0:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]

    # 构造特征编码，标签，采样时间的元组
    train_set = (train_set_x, train_set_y, train_set_t)
    valid_set = (valid_set_x, valid_set_y, valid_set_t)
    test_set = (test_set_x, test_set_y, test_set_t)

    return train_set, valid_set, test_set

'''
梯度下降
    :param
        tparams：网络连接参数
        grads：偏导
        x：网络的输入
        t：网络的时间维信息输入
        mask：序列掩膜
        y：标签
        cost：损失函数
    :return
        f_grad_shared：网络权值参数梯度以及梯度的期望
        f_update：网络权值参数
'''
def adadelta(tparams, grads, x, t, mask, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in
                      tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, t, mask, y], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in
             zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

    return f_grad_shared, f_update

'''
AUC计算函数
    :param
        test_model：模型函数（输入为(x,t,mask)，输出为p_y_given_x）
        datasets：数据集（验证集或测试集）
    :return
        auc：模型在数据集datasets上的AUC值
'''
def calculate_auc(test_model, datasets):
    batchSize = 10      # 指定datasets的batchSize
    n_batches = int(np.ceil(float(len(datasets[0])) / float(batchSize)))    # 批次数
    scoreVec = []
    for index in xrange(n_batches):
        x, t, mask = padMatrix(datasets[0][index * batchSize:(index + 1) * batchSize],
                               datasets[2][index * batchSize:(index + 1) * batchSize])
        scoreVec.extend(list(test_model(x, t, mask).flatten()))     # 将函数输出整合成列表形式，加入到scoreVec列表里
    labels = datasets[1]        # 取标签
    auc = roc_auc_score(list(labels), list(scoreVec))   # 二分类，注意labels中的类别数
    return auc

'''
序列填充
将短序列的后续序列，以零填充
    :param
        seqs：输入序列
        times：时间序列
    :return
        x：填充后的输入序列
        t：填充后的时间序列
        x_mask：序列掩膜
'''
def padMatrix(seqs, times):
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int32')
    t = np.zeros((maxlen, n_samples)).astype(config.floatX)
    x_mask = np.zeros((maxlen, n_samples)).astype(config.floatX)
    for idx, (seq, time) in enumerate(zip(seqs, times)):
        x[:lengths[idx], idx] = seq
        t[:lengths[idx], idx] = time
        x_mask[:lengths[idx], idx] = 1.
    t = np.log(t + 1.)      # 时间间隔对数化

    return x, t, x_mask


def train_GRU_RNN(
        dataFile='data.txt',
        labelFile='label.txt',
        timeFile='',
        embFile='emb.txt',
        outFile='out.txt',
        inputDimSize=100,
        embDimSize=100,
        hiddenDimSize=100,
        max_epochs=100,
        L2_reg=0.,
        batchSize=100,
        use_dropout=True
):
    options = locals().copy()   # 变量加载到本地

    print 'Loading data ... ',
    trainSet, validSet, testSet = load_data(dataFile, labelFile, timeFile=timeFile)     # 加载数据并重新整合
    n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))        # 训练集批次数
    print 'done!!'

    print 'Building the model ... ',
    params = init_params(options)       # 初始化网络连接参数
    tparams = init_tparams(params)      # 将网络连接参数设为符号变量
    Wemb = theano.shared(params['W_emb'], name='W_emb')     # 将输入设为符号变量
    use_noise, x, t, mask, y, p_y_given_x, cost = build_model(tparams, options, Wemb)   # 构造网络模型
    print 'done!!'

    print 'Constructing the optimizer ... ',
    grads = T.grad(cost, wrt=tparams.values())      # 求损失函数偏导
    f_grad_shared, f_update = adadelta(tparams, grads, x, t, mask, y, cost)     # 求参数梯度函数与参数函数
    print 'done!!'

    test_model = theano.function(inputs=[x, t, mask], outputs=p_y_given_x, name='test_model')   # 拟合函数

    bestValidAuc = 0.
    bestTestAuc = 0.
    iteration = 0
    bestParams = OrderedDict()
    print 'Optimization start !!'
    for epoch in xrange(max_epochs):
        for index in random.sample(range(n_batches), n_batches):
            use_noise.set_value(1.)     # 训练状态置1
            # 补0
            x, t, mask = padMatrix(trainSet[0][index * batchSize:(index + 1) * batchSize],
                                   trainSet[2][index * batchSize:(index + 1) * batchSize])
            y = trainSet[1][index * batchSize:(index + 1) * batchSize]
            cost = f_grad_shared(x, t, mask, y)     # 求损失
            f_update()  # 更新参数
            iteration += 1

        use_noise.set_value(0.)     # 验证测试状态置0
        validAuc = calculate_auc(test_model, validSet)      # 计算验证集AUC
        print 'epoch:%d, valid_auc:%f' % (epoch, validAuc)
        if (validAuc > bestValidAuc):
            bestValidAuc = validAuc
            testAuc = calculate_auc(test_model, testSet)
            bestTestAuc = testAuc
            bestParams = unzip(tparams)
            print 'Currenlty the best test_auc:%f' % testAuc

    np.savez_compressed(outFile, **bestParams)


if __name__ == '__main__':
    '''
    dataFile = sys.argv[1]
    timeFile = sys.argv[2]
    labelFile = sys.argv[3]
    embFile = sys.argv[4]
    outFile = sys.argv[5]
    '''
    dataFile = 'sequences.pkl'
    timeFile = 'times.pkl'
    labelFile = 'labels.pkl'
    embFile = 'emb.pkl'
    outFile = 'out.csv'

    inputDimSize = 100  # The number of unique medical codes
    embDimSize = 100  # The size of the code embedding
    hiddenDimSize = 100  # The size of the hidden layer of the GRU
    max_epochs = 100  # Maximum epochs to train
    L2_reg = 0.001  # L2 regularization for the logistic weight
    batchSize = 10  # The size of the mini-batch
    use_dropout = True  # Whether to use a dropout between the GRU and the logistic layer

    train_GRU_RNN(dataFile=dataFile, labelFile=labelFile, timeFile=timeFile, embFile=embFile, outFile=outFile,
                  inputDimSize=inputDimSize, embDimSize=embDimSize, hiddenDimSize=hiddenDimSize, max_epochs=max_epochs,
                  L2_reg=L2_reg, batchSize=batchSize, use_dropout=use_dropout)
