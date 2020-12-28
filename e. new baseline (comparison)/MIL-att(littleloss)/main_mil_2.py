# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:54:18 2020

@author: DELL
"""

#x_mask作用？？？
#attention_v2变成attention_v1
#embeddings 改为embedding
#word_embedding = 100


import os
from collections import defaultdict
import numpy 
numpy.random.seed(1)
import tensorflow as tf
import logging
import math
from tensorflow import logging  as log
from tensorflow.python import debug as tf_debug
from collections import OrderedDict
from data_iteratorMIL import TextIterator
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers
import warnings
import pickle as pkl
import sys
import pprint
import pdb
import os
import copy
import time
import matplotlib.pyplot as plt
import pickle

logger = logging.getLogger(__name__)


def _s(pp, name):  # add perfix
    return '{}_{}'.format(pp, name)


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('{} is not in the archive'.format(kk))
            continue
        params[kk] = pp[kk]

    return params


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * numpy.sqrt(6.0 / (fan_in + fan_out))
    high = constant * numpy.sqrt(6.0 / (fan_in + fan_out))
    W = numpy.random.uniform(low=low, high=high, size=(fan_in, fan_out))
    return W.astype('float32')


def ortho_weight(ndim):  # used by norm_weight below
    """
    Random orthogonal weights
    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        # W = numpy.random.uniform(-0.5,0.5,size=(nin,nout))
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def attention_v1(input, masks, name='attention', nin=100, keep=1.0, is_training=True):
    # input is batch,time_step,hidden_state (32*40)*13*600   mask (32*40)*13
    # hidden layer is:batch,hidden_shape,attention_hidden_size (32*40)*13*1200 or (32*40)*13*600
    # attention shape after squeeze is (32*40)*13, # batch,time_step,attention_size (32*40)*13*1
    with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
        hidden = tf.layers.dense(input, nin / 2, activation=tf.nn.tanh, use_bias=True,
                                kernel_initializer=layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                name='hidden', reuse=tf.AUTO_REUSE)
        attention = tf.layers.dense(hidden, 1, activation=None, use_bias=False,
                                    kernel_initializer=layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                    name='out',
                                    reuse=tf.AUTO_REUSE)
        padding = tf.fill(tf.shape(attention), float('-1e8'))  # float('-inf')
        attention = tf.where(tf.equal(tf.expand_dims(masks, -1), 0.), padding,
                            attention)  # fill 0 with a small number for softmax
        attention = tf.nn.softmax(attention, 1,name='softmax') * tf.expand_dims(masks,
                                                                -1)  # (32*40)*13*-1
        results = tf.reduce_sum(input * attention, axis=1)  # (32*40)*-1 is not really neccesary,
    return results #(32*40)*nin



def lstm_filter(input, mask, keep_prob, prefix='lstm', dim=50, is_training=True):
    with tf.variable_scope(name_or_scope=prefix, reuse=tf.AUTO_REUSE):
        sequence = tf.cast(tf.reduce_sum(mask, 1), tf.int32)
        lstm_fw_cell = rnn.LSTMCell(dim, forget_bias=0.0, initializer=tf.orthogonal_initializer(), state_is_tuple=True)
        keep_rate = tf.cond(is_training is not False and keep_prob < 1, lambda: 0.8, lambda: 1.0)
        cell_dp_fw = rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=keep_rate)
        outputs, _ = tf.nn.dynamic_rnn(cell_dp_fw, input, sequence_length=sequence,swap_memory=False,
                                       dtype=tf.float32)
    return outputs 


def bilstm_filter(input, mask, keep_prob, prefix='lstm', dim = 50, is_training=True):
    with tf.variable_scope(name_or_scope=prefix, reuse=tf.AUTO_REUSE):
        sequence = tf.cast(tf.reduce_sum(mask, 1), tf.int32)
        lstm_fw_cell = rnn.LSTMBlockCell(dim, forget_bias=1.0)# initializer=tf.orthogonal_initializer(), state_is_tuple=True
        # back directions
        lstm_bw_cell = rnn.LSTMBlockCell(dim, forget_bias=1.0)
        keep_rate = tf.cond(is_training is not False and keep_prob < 1, lambda: 0.8, lambda: 1.0)
        cell_dp_fw = rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=keep_rate)
        cell_dp_bw = rnn.DropoutWrapper(cell=lstm_bw_cell, output_keep_prob=keep_rate)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_dp_fw, cell_dp_bw, input, sequence_length=sequence,swap_memory=False,
                                                     dtype=tf.float32)  # batch major
    return outputs 
   # input shape: [batch*new_s, word, options['dim_word']  (32*40)*12*100
   # output shape: batch*news,sequence,2*lstm_units  (32*40)*12*600(nin)


def init_params(options, worddicts):
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    # read embedding from GloVe
    if options['embedding']:
        with open(options['embedding'], 'r') as f:
            for line in f:
                tmp = line.split()
                word = tmp[0]
                vector = tmp[1:]
                if word in worddicts and worddicts[word] < options['n_words']:
                    try:
                        params['Wemb'][worddicts[word], :] = vector
                        # encoder: bidirectional RNN
                    except ValueError as e:
                        print(str(e))
    return params


def word_embedding(options, params):
    embedding = tf.get_variable("embedding", shape=[options['n_words'], options['dim_word']],
                                 initializer=tf.constant_initializer(numpy.array(
                                     params['Wemb'])))  # tf.constant_initializer(numpy.array(params['Wemb']))
    return embedding


# (32, 11, 16) (32, 11) (32,)
def prepare_data(sequence, labels, options, maxlen=None, max_word=100):
    # length = [len(s) for s in sequence]
    length = []   #一天里面 number of news
    for i in sequence: #x,one batch里面one day
        length.append(len(i))
    if maxlen is not None:  # max length is the news level
        new_sequence = []
        new_lengths = []
        
        for l, s in zip(length, sequence):
            if l < maxlen:
                new_sequence.append(s)
                new_lengths.append(l)

        length = new_lengths  # This step is to filter the sentence which length is bigger
        sequence = new_sequence  # than the max length. length means number of news. sequence means 
        ##TODO need to be careful, set the max length bigger to avoid bug
        if len(length) < 1:
            return None, None, None, None, None, None, None, None
        
    maxlen_x = numpy.max(length)  # max time step   <100(maxlen)
    n_samples = len(sequence)  # number of samples == batch
    max_sequence = max(len(j) for i in sequence for j in i)  # find the sequence max length
    max_sequence = max_word if max_sequence > max_word else max_sequence  # shrink the data size
    ##TODO for x
    x = numpy.zeros((n_samples, maxlen_x, max_sequence)).astype('int64')  #(32, 11, 16)
    x_mask = numpy.zeros((n_samples, maxlen_x)).astype('float32')  #(32, 11)
    ##TODO for label
    l = numpy.zeros((n_samples,)).astype('int64')
    for index, (i, ll) in enumerate(zip(sequence, labels)):  # batch size
        l[index] = ll
        for idx, ss in enumerate(i):  # time step
            # x[idx, index, :sequence_length[idx]] = ss
            if len(ss) < max_sequence:
                x[index, idx, :len(ss)] = ss
            else:
                x[index, idx, :max_sequence] = ss[:max_sequence]
            x_mask[index, idx] = 1.

    #################3 maxlen_x 都小于 maxlen(100),但每一组maxlen_x都不一样！！
    return x, x_mask, l

def Euclidean_distance(x):  
    "x.shape: (32,N,emb_size), then the function is to get a batch matrix (32,N,N)"
    #N = x.get_shape().as_list()[1]
    N = tf.shape(x)[1]
    # expand dim
    x_expand = tf.expand_dims(x,2) #32,N,1,emb_size
    xT_expand = tf.expand_dims(x,3) #32,N,emb_size,1
    # tf.tile--> make copy
    x_tile = tf.tile(x_expand,[1,1,N,1])  #32,N,N,emb_size
    xT_tile = tf.tile(xT_expand,[1,1,1,N])  #32,N,emb_size,N
    # transpose
    l = tf.transpose(xT_tile,[0,3,1,2])  #32,N,N,emb_size
    ## Calculate the distance
    eu_dis = tf.reduce_sum(tf.square(x_tile-l),3)  #32,N,N
    kernel_dis = tf.exp(-eu_dis)  #values between(0,1)
    
    return kernel_dis

def instance_diff(l):
    "The shape of input x is:(32,N),with values between(0,1), the shape of output (instance_pred(i)-instance_pred(j))^2 is (32,N,N)"
    #N = l.get_shape().as_list()[-1]
    N = tf.shape(l)[1]
    l_expand = tf.expand_dims(l,1)  #32,1,N
    lT_expand = tf.expand_dims(l,2) #32,N,1
    l_tile = tf.tile(l_expand,[1,N,1])
    lT_tile = tf.tile(lT_expand,[1,1,N])
    dist = tf.square(l_tile-lT_tile)   #32,N,N
    
    return dist


def build_model(embedding, options):
    """ Builds the entire computational graph used for training
    """
    # description string: #words x #samples
    with tf.device('/gpu:0'):
        with tf.variable_scope('input'):
            x = tf.placeholder(tf.int64, shape=[None, None, None],
                               name='x')  # 3D vector batch,N and instances(before embedding)40*32*13
            x_mask = tf.placeholder(tf.float32, shape=[None, None], name='x_mask')  # mask batch,N
            y = tf.placeholder(tf.int64, shape=[None], name='y') #group actual
            ##TODO important    
            keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
            is_training = tf.placeholder(tf.bool, name='is_training')
            alpha_balance = tf.placeholder(tf.float32,[],name = 'alpha_balance')
            ##TODO important
            sequence_mask = tf.cast(tf.abs(tf.sign(x)), tf.float32)  # 3D
            n_timesteps = tf.shape(x)[0]  # time steps
            ##TODO word embedding
            emb = tf.nn.embedding_lookup(embedding, x)
            
    with tf.device('/gpu:0'):
        # fed into the input of BILSTM from the official document
        with tf.name_scope('instance_embedding'):
            batch = tf.shape(emb)[0] #32
            N = tf.shape(emb)[1] #40 N instances in a group
            word = tf.shape(emb)[2]  #13
            ##TODO make instances prediction through attention encoding and MLP
            with tf.variable_scope(name_or_scope='instance_embedding', reuse=tf.AUTO_REUSE):
                word_level_inputs = tf.reshape(emb, [batch * N, word, options['dim_word']])
                word_level_mask = tf.reshape(sequence_mask, [batch * N, word])
                ##TODO word level LSTM
                word_encoder_out = bilstm_filter(word_level_inputs, word_level_mask, keep_prob,prefix='sequence_encode', dim=options['dim'],is_training=is_training)  # output shape: batch*news,sequence,2*lstm_units(32*40)*12*600
                word_encoder_out = tf.concat(word_encoder_out, 2) * tf.expand_dims(word_level_mask, -1)  
                ################################### TODO word-attention
                word_level_output = attention_v1(word_encoder_out, word_level_mask, name='word_attention', keep=keep_prob,is_training=is_training)
            
                if options['use_dropout']:
                    word_level_output = layers.dropout(word_level_output, keep_prob=keep_prob, is_training=is_training,seed=None)
                #32*N,D
            
                att = tf.reshape(word_level_output, [batch, N, 2*options['dim']])
                ##TODO att shape 32*40*600
        with tf.name_scope('instance_prediction'):
            logit = tf.layers.dense(word_level_output, 150,activation=tf.nn.tanh,use_bias=True,kernel_initializer=layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32),name='inst_temp', reuse=tf.AUTO_REUSE)
            if options['use_dropout']:
                logit = layers.dropout(logit, keep_prob=keep_prob, is_training=is_training,seed=None)
            
            pred_sig_ = tf.layers.dense(logit, 1, activation=None, use_bias=True,kernel_initializer=layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),name='inst_pred', reuse=tf.AUTO_REUSE)
            inst_pred = tf.nn.sigmoid(pred_sig_)#32*N,1, float32
            L = tf.reshape(inst_pred,[batch,N])  #x_mask??
            ##TODO make group prediction through average instance predictions
            group_ = tf.reduce_sum(L,1)/tf.cast(N,tf.float32)  # Do the instance_pred average  32,
            group_pred = tf.cast(tf.ceil(group_-0.5),tf.int64)    #32,
            
            
            ##TODO new cost
            logger.info('Building f_cost...')
            x_simil = Euclidean_distance(att)  #32,N,N   有placeholder
            l_diff = instance_diff(L) #32,N,N   有placeholder
            simil_cost = tf.reduce_sum(tf.multiply(x_simil,l_diff),[1,2])/tf.cast(N*N,tf.float32)  #32,
            group_cost = tf.cast(tf.square(y-group_pred),tf.float32)  #y:32,  group_pred???
            # cost由int64变为float32
            total_cost = simil_cost + alpha_balance * group_cost  #[32,1]
            cost = tf.reshape(total_cost,(1,-1))  #1,32
            
            
            """pred = tf.layers.dense(logit, 2, activation=None, use_bias=True,
                                kernel_initializer=layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                name='fout', reuse=tf.AUTO_REUSE)#32,2
            labels = tf.one_hot(y, depth=2, axis=1)#32,2
            preds = tf.nn.softmax(pred, 1,name='softmax')  #32,2
            cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels)  #1,32"""

        logger.info('Done')


        with tf.variable_scope('logging'):
            tf.summary.scalar('current_cost', tf.reduce_mean(cost))
            tf.summary.histogram('predicted_value', group_pred)
            summary = tf.summary.merge_all()

    return is_training, cost, x, x_mask, y, n_timesteps, group_pred, summary  
    #return is_training, cost, x, x_mask, y, n_timesteps, preds, summary
    # preds:(32,2)  (0.2228,1-0.228)
  
    
#valid_acc, valid_loss,valid_final_result = predict_pro_acc(sess, cost, prepare_data, model_options, valid, maxlen,
                                                            #correct_pred, pred, summary, eidx, is_training, train_op,loss_plot,
                                                            #validate_writer,validate=True)
def predict_pro_acc(sess, cost, prepare_data, model_options, iterator, maxlen, correct_pred, group_pred, summary, eidx,
                    is_training,train_op, plot=None,writer=None,validate=False):
    #pred-->group_pred
    num = 0
    valid_acc = 0
    total_cost = 0
    loss = 0
    result = 0
    final_result=[]
    #sess.add_tensor_filter("val_test_spot")
    for x_sent, y_sent in iterator:
        num += len(x_sent)
        data_x, data_x_mask, data_y = prepare_data(x_sent,y_sent,model_options,maxlen=maxlen)
        print(data_x.shape, data_x_mask.shape, data_y.shape)
        loss, result, preds = sess.run([cost, correct_pred, group_pred],  #pred(32,2)-->group_pred(32,)
                                       feed_dict={'input/x:0': data_x, 'input/x_mask:0': data_x_mask,
                                                  'input/y:0': data_y, 
                                                  'input/keep_prob:0': 1.0,
                                                  'input/is_training:0': is_training,
                                                  'input/alpha_balance:0':1.0})
        valid_acc += result.sum()
        total_cost += loss.sum()
        if plot is not None:
            if validate is True:
                plot['validate'].append(loss.sum()/len(x_sent))
            else:
                plot['testing'].append(loss.sum()/len(x_sent))
    final_result.extend(result.tolist())
    final_acc = 1.0 * valid_acc / num
    final_loss = 1.0 * total_cost / num
    # if writer is not None:
    #    writer.add_summary(test_summary, eidx)

    print(preds, result, num)

    return final_acc, final_loss, final_result


def train(
        dim_word = 100,  # word vector dimensionality
        dim = 50, #100,  # the number of GRU units
        encoder='lstm',  # encoder model
        decoder='lstm',  # decoder model
        patience=10,  # early stopping patience
        max_epochs=300,
        finish_after=10000000,  # finish after this many updates
        decay_c=0.,  # L2 regularization penalty
        clip_c=-1.,  # gradient clipping threshold
        lrate=0.0004,  # learning rate
        alpha_balance = 0.04,
        n_words=100000,  # vocabulary size
        n_words_lemma=100000,
        maxlen=100,  # maximum length of the description
        optimizer='adam',
        batch_size=32,
        valid_batch_size=32,
        save_model='results/mil_att/',
        saveto='MIL_ATT.npz',
        dispFreq=100,
        validFreq=1000,
        saveFreq=1000,  # save the parameters after every saveFreq updates
        use_dropout=False,
        reload_=False,
        verbose=False,  # print verbose information for debug but slow speed
        types='title',
        cut_word=False,
        cut_news=False,
        keep_prob = 0.8,
        datasets=[],
        valid_datasets=[],
        test_datasets=[],
        dictionary=[],
        embedding='',  # pretrain embedding file, such as word2vec, GLOVE
        RUN_NAME="histogram_visualization",
        wait_N=10
):
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s",
                        filename='results/mil_att/log_result.txt')
    # Model options
    model_options = locals().copy()

    with open(dictionary, 'rb') as f:
        worddicts = pkl.load(f)

    logger.info("Loading knowledge base ...")

    # reload options
    """if reload_ and os.path.exists(saveto):
        logger.info("Reload options")
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)"""

    logger.debug(pprint.pformat(model_options))
    
    logger.info("Loading data")
    train = TextIterator(datasets[0], datasets[1],
                         dict=dictionary,
                         types=types,
                         n_words=n_words,
                         batch_size=batch_size,
                         cut_word=cut_word,
                         cut_news=cut_news,
                         shuffle=True, shuffle_sentence=False,quiet=False)
    train_valid = TextIterator(datasets[0], datasets[1],
                               dict=dictionary,
                               types=types,
                               n_words=n_words,
                               batch_size=valid_batch_size,
                               cut_word=cut_word,
                               cut_news=cut_news,
                               shuffle=False, shuffle_sentence=False,quiet=False)
    valid = TextIterator(valid_datasets[0], valid_datasets[1],
                         dict=dictionary,
                         types=types,
                         n_words=n_words,
                         batch_size=valid_batch_size,
                         cut_word=cut_word,
                         cut_news=cut_news,
                         shuffle=True, shuffle_sentence=False,quiet=False)
    test = TextIterator(test_datasets[0], test_datasets[1],
                        dict=dictionary,
                        types=types,
                        n_words=n_words,
                        batch_size=valid_batch_size,
                        cut_word=cut_word,
                        cut_news=cut_news,
                        shuffle=True, shuffle_sentence=False,quiet=False)

    # Initialize (or reload) the parameters using 'model_options'
    # then build the tensorflow graph
    logger.info("init_word_embedding")
    params = init_params(model_options, worddicts)
    embedding = word_embedding(model_options, params)
    is_training, cost, x, x_mask, y, n_timesteps, group_pred, summary = build_model(embedding, model_options)
    #is_training, cost, x, x_mask, y, n_timesteps, pred, summary = build_model(embedding, model_options)
    with tf.variable_scope('train'):
        lr = tf.Variable(0.0, trainable=False)

        def assign_lr(session, lr_value):
            session.run(tf.assign(lr, lr_value))

        logger.info('Building optimizers...')
        #optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr,rho=0.95)
        logger.info('Done')
        # print all variables
        tvars = tf.trainable_variables()
        for var in tvars:
            print(var.name, var.shape)
        lossL = tf.add_n([tf.nn.l2_loss(v) for v in tvars if ('embedding' not in v.name and 'bias' not in v.name)])#
        lossL2=lossL * 0.0005
        print("don't do L2 variables:")
        print([v.name for v in tvars if ('embedding' in v.name or 'bias' in v.name)])
        print("\n do L2 variables:")
        print([v.name for v in tvars if ('embedding' not in v.name and 'bias' not in v.name)])
        cost = cost + lossL2
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), model_options['clip_c'])
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.apply_gradients(zip(grads, tvars))
        # train_op = optimizer.minimize(cost)
        op_loss = tf.reduce_mean(cost)
        op_L2 = tf.reduce_mean(lossL)
        logger.info("correct_pred")
        correct_pred = tf.equal(group_pred, y)  # make prediction
        #correct_pred = tf.equal(tf.argmax(input=pred, axis=1), y)  # make prediction
        logger.info("Done")

        temp_accuracy = tf.cast(correct_pred, tf.float32)  # change to float32

    logger.info("init variables")
    init = tf.global_variables_initializer()
    logger.info("Done")
    # saver
    saver = tf.train.Saver(max_to_keep=15)

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True  
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        training_writer = tf.summary.FileWriter("results/mil_att/logs/{}/training".format(RUN_NAME), sess.graph)
        validate_writer = tf.summary.FileWriter("results/mil_att/logs/{}/validate".format(RUN_NAME), sess.graph)
        testing_writer = tf.summary.FileWriter("results/mil_att/logs/{}/testing".format(RUN_NAME), sess.graph)
        sess.run(init)
        history_errs = []
        history_valid_result = []
        history_test_result = []
        # reload history
        """if reload_ and os.path.exists(saveto):
            logger.info("Reload history error")
            history_errs = list(numpy.load(saveto)['history_errs'])"""

        bad_counter = 0

        if validFreq == -1:
            validFreq = len(train[0]) / batch_size
        if saveFreq == -1:
            saveFreq = len(train[0]) / batch_size
        
        loss_plot=defaultdict(list)
        uidx = 0
        estop = False
        best_num = -1
        best_epoch_num = 0
        lr_change_list = []
        wait_counter = 0
        wait_N = model_options['wait_N']
        learning_rate = model_options['lrate']
        alpha_balance = model_options['alpha_balance']
        assign_lr(sess, learning_rate)
        for eidx in range(max_epochs):
            n_samples = 0
            training_cost = 0
            training_acc = 0
            for x,y in train:
                n_samples += len(x)
                uidx += 1
                keep_prob = model_options['keep_prob']
                is_training = True
                data_x, data_x_mask,data_y = prepare_data(x,y,model_options,maxlen=maxlen)
                print(data_x.shape, data_x_mask.shape, data_y.shape)
                assert data_y.shape[0] == data_x.shape[0], 'Size does not match'
                if x is None:
                    logger.debug('Minibatch with zero sample under length {0}'.format(maxlen))
                    uidx -= 1
                    continue
                ud_start = time.time()
                _, loss,loss_no_mean,temp_acc,l2_check = sess.run([train_op, op_loss,cost,temp_accuracy,op_L2],
                                   feed_dict={'input/x:0': data_x, 'input/x_mask:0': data_x_mask, 'input/y:0': data_y,
                                              'input/keep_prob:0': keep_prob, 'input/is_training:0': is_training,'input/alpha_balance:0': alpha_balance})
                ud = time.time() - ud_start
                training_cost += loss_no_mean.sum()
                training_acc += temp_acc.sum()
                loss_plot['training'].append(loss)
                if numpy.mod(uidx, dispFreq) == 0:
                    logger.debug('Epoch {0} Update {1} Cost {2} L2 {3} TIME {4}'.format(eidx, uidx, loss,l2_check,ud))

                # validate model on validation set and early stop if necessary
                if numpy.mod(uidx, validFreq) == 0:
                    is_training = False
                    
                    #pred-->group_pred
                    valid_acc, valid_loss,valid_final_result = predict_pro_acc(sess, cost, prepare_data, model_options, valid, maxlen,
                                                            correct_pred, group_pred, summary, eidx, is_training, train_op,loss_plot,
                                                            validate_writer,validate=True)
                    test_acc, test_loss,test_final_result = predict_pro_acc(sess, cost, prepare_data, model_options, test, maxlen,
                                                          correct_pred, group_pred, summary, eidx, is_training, train_op,loss_plot,
                                                          testing_writer)
                    # valid_err = 1.0 - valid_acc
                    valid_err = valid_loss
                    history_errs.append(valid_err)
                    history_valid_result.append(valid_final_result)
                    history_test_result.append(test_final_result)
                    loss_plot['validate_ep'].append(valid_loss)
                    loss_plot['val_ac'].append(valid_acc)
                    loss_plot['testing_ep'].append(test_loss)
                    loss_plot['test_ac'].append(test_acc)
                    logger.debug('Epoch  {0}'.format(eidx))
                    logger.debug('Valid cost  {0}'.format(valid_loss))
                    logger.debug('Valid accuracy  {0}'.format(valid_acc))
                    logger.debug('Test cost  {0}'.format(test_loss))
                    logger.debug('Test accuracy  {0}'.format(test_acc))
                    logger.debug('learning_rate:  {0}'.format(learning_rate))
                    

                    if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                        best_num = best_num + 1
                        best_epoch_num = eidx
                        wait_counter = 0
                        #logger.info("Saving...")
                        #saver.save(sess, _s(_s(_s(save_model, "epoch"), str(best_num)), "model.ckpt"))
                        #logger.info(_s(_s(_s(save_model, "epoch"), str(best_num)), "model.ckpt"))
                        #numpy.savez(saveto, history_errs=history_errs, **params)
                        #pkl.dump(model_options, open('{}.pkl'.format(saveto), 'wb'))
                        #logger.info("Done")

                    if valid_err > numpy.array(history_errs).min():
                        wait_counter += 1
                    # wait_counter +=1 if valid_err>numpy.array(history_errs).min() else 0
                    if wait_counter >= wait_N:
                        logger.info("wait_counter max, need to half the lr")
                        # print 'wait_counter max, need to half the lr'
                        bad_counter += 1
                        wait_counter = 0
                        logger.debug('bad_counter:  {0}'.format(bad_counter))
                        # TODO change the learining rate
                        learning_rate = learning_rate * 0.9
                        # learning_rate = learning_rate
                        assign_lr(sess, learning_rate)
                        lr_change_list.append(eidx)
                        logger.debug('lrate change to:   {0}'.format(learning_rate))
                        # print 'lrate change to: ' + str(lrate)

                    if bad_counter > patience:
                        logger.info("Early Stop!")
                        estop = True
                        break

                    if numpy.isnan(valid_err):
                        pdb.set_trace()

                        # finish after this many updates
                if uidx >= finish_after:
                    logger.debug('Finishing after iterations!  {0}'.format(uidx))
                    # print 'Finishing after %d iterations!' % uidx
                    estop = True
                    break
            logger.debug('Seen samples:  {0}'.format(n_samples))
            logger.debug('Training accuracy:  {0}'.format(1.0 * training_acc/n_samples))
            loss_plot['training_ep'].append(training_cost/n_samples)
            loss_plot['train_ep'].append(training_acc/n_samples)
            # print 'Seen %d samples' % n_samples
            logger.debug('results/mil_att/Saved loss_plot pickle')
            with open("results/mil_att/important_plot.pickle",'wb') as handle:
                pkl.dump(loss_plot, handle, protocol=pkl.HIGHEST_PROTOCOL)
            if estop:
                break

    """with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        # Restore variables from disk.
        saver.restore(sess, _s(_s(_s(save_model, "epoch"), str(best_num)), "model.ckpt"))
        keep_prob = 1
        is_training = False
        #alpha_balance = 1
        logger.info('=' * 80)
        logger.info('Final Result')
        logger.info('=' * 80)
        logger.debug('best epoch   {0}'.format(best_epoch_num))

        #pred-->group_pred
        valid_acc, valid_cost,valid_final_result = predict_pro_acc(sess, cost, prepare_data, model_options, valid,
                                                maxlen, correct_pred, group_pred, summary, eidx,train_op, is_training, None)
        logger.debug('Valid cost   {0}'.format(valid_cost))
        logger.debug('Valid accuracy   {0}'.format(valid_acc))

        # print 'Valid cost', valid_cost
        # print 'Valid accuracy', valid_acc

        test_acc, test_cost,test_final_result = predict_pro_acc(sess, cost, prepare_data, model_options, test,
                                              maxlen, correct_pred, group_pred, summary, eidx,train_op, is_training, None)
        logger.debug('Test cost   {0}'.format(test_cost))
        logger.debug('Test accuracy   {0}'.format(test_acc))

        # print 'best epoch ', best_epoch_num
        train_acc, train_cost,_ = predict_pro_acc(sess, cost, prepare_data, model_options, train_valid,
                                                maxlen, correct_pred, group_pred, summary, eidx,train_op, is_training, None)
        logger.debug('Train cost   {0}'.format(train_cost))
        logger.debug('Train accuracy   {0}'.format(train_acc))
        valid_m=numpy.array(history_valid_result)
        test_m=numpy.array(history_test_result)
        valid_final_result = (numpy.array([valid_final_result])==False)
        test_final_result = (numpy.array([test_final_result])==False)
        #print(numpy.all(valid_m, axis = 0))
        #print(numpy.all(test_m, axis=0))
        print('validation: all prediction through every epoch that are the same:',numpy.where(numpy.all(valid_m, axis = 0)))
        print('testing: all prediction through every epoch that are the same:',numpy.where(numpy.all(test_m, axis=0)))
        print('validation: final prediction that is False:',numpy.where(valid_final_result))
        print('testing: final prediction that is False:',numpy.where(test_final_result))
        if os.path.exists('results/mil_att/history_predict.npz'):
            logger.info("Load and save to history_predict.npz")
            valid_history = numpy.load('results/mil_att/history_predict.npz')['valid_final_result']
            test_history = numpy.load('results/mil_att/history_predict.npz')['test_final_result']
            vv=numpy.concatenate((valid_history,valid_final_result),axis=0)
            tt=numpy.concatenate((test_history,valid_final_result),axis=0)
            print('Concate shape valid:',vv.shape)
            print('Print all validate history outputs that return False',numpy.where(numpy.all(vv,axis=0)))
            print('Concate shape test:',tt.shape)
            print('Print all test history outputs that return False',numpy.where(numpy.all(tt,axis=0)))
            numpy.savez('results/mil_att/history_predict.npz',valid_final_result=vv,test_final_result=tt,**params)
        else:
            numpy.savez('results/mil_att/history_predict.npz',valid_final_result=valid_final_result,test_final_result=test_final_result,**params)
        # print 'Train cost', train_cost
        # print 'Train accuracy', train_acc

        # print 'Test cost   ', test_cost
        # print 'Test accuracy   ', test_acc

        return None"""

    
if __name__ == '__main__':
    pass    