#!/usr/bin/env python
# Author: YJHMITWEB
# E-mail: yjhmitweb@gmail.com

import numpy as np
import os
import time
import tensorflow as tf
from utils.common_ops import Conv1D, RNN, BN, Dense
from configs.config import c as config
from utils.generator import generator_valid, generator_train

class GLoMo():
    def __init__(self, config):
        self.batch_size = config.BATCH_SIZE
        self.embedding_dims = config.EMBEDDING_DIMS
        self.learning_rate = config.LEARNING_RATE
        self.momentum = config.MOMENTUM
        self.optimizer = config.DEFAULT_OPTIMIZER
        self.epochs = config.TRAINING_EPOCH
        self.iter_per_epo = config.ITERATION_PER_EPOCH
        self.ckpt_dir = config.CHECKPOINT_DIR
        self.log_dir = config.LOG_DIR
        self.num_classes = config.NUM_CLASSES
        self.evaluation_per_epoch = config.EVALUATION_PER_EPOCH
        self.evaluation_runs = config.EVALUATION_RUNS
        self.sentence_length = config.SENTENCE_LENGTH
        self.vocab_size = config.VOCAB_SIZE

        self.key_cnn_layers = config.KEY_CNN_LAYERS
        self.query_cnn_layers = config.QUERY_CNN_LAYERS
        # self.graph_layers = list(set(sorted(config.GRAPH_LAYERS)))
        self.graph_layers = 'every'
        self.key_cnn_conv_length = config.KEY_CNN_CONV_LENGTH
        self.query_cnn_conv_length = config.QUERY_CNN_CONV_LENGTH
        self.graph_bias = config.GRAPH_BIAS
        self.graph_w_units = config.GRAPH_W_UNITS
        self.graph_scope = config.GRAPH_SCOPE

        self.rnn_layers = config.RNN_EACH_LAYERS
        # self.using_graph_at = list(set(sorted(config.USING_GRAPH_AT)))
        self.using_graph_at = 'every'
        self.feature_scope = config.FEATURE_SCOPE
        self.using_rnn_at_each_graph = config.USING_RNN_AT_EACH_GRAPH
        self.linear_shortcut_rate = config.LINEAR_SHORTCUT_RATE
        self.linear_activation = config.LINEAR_ACTIVATION
        self.linear_bias = config.LINEAR_BIAS
        self.prediction_length = config.PREDICTION_LENGTH
        ### Net inputs, shared by both feature predictor and graph predictor
        ### Assuming has shape: [batch, num_steps, channels]
        # self.inputs = tf.placeholder(
        #     dtype=tf.int32, shape=(None, self.sentence_length, self.embedding_dims))
        self.inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.sentence_length))
        # embeding matrix of words
        self.embedding_matrix = np.random.random([self.vocab_size, self.embedding_dims]).astype(np.float32)
        
        self.embedding = tf.reshape(tf.nn.embedding_lookup(self.embedding_matrix, self.inputs, name='embedding_layer_graph'), [self.batch_size, self.sentence_length, self.embedding_dims])

    def Calculate_Graph(self, key_feature, query_feature, idx):
        """
        First we squeeze the added dimension for conv ops which is the second axis.
        Then, we transpose key_feature then dot multiplying it with key feature.
        :param key_feature: must have the same shape as query feature.
                            e.g. [batch_size, num_steps, channels]
        :param query_feature: e.g. [batch_size, num_steps, channels]
        :return: relation graph: [batch_size, channels, channels]
        """
        
        W_k = tf.Variable(tf.ones([self.sentence_length, 128],dtype=tf.float32), name='W_key_{}'.format(idx))
        W_q = tf.Variable(tf.ones([self.sentence_length, 128],dtype=tf.float32), name='W_query_{}'.format(idx))

        key_feature = tf.einsum('ijk,jk->ijk', key_feature, W_k, name = 'key_feature_matmul_{}'.format(idx))
        query_feature = tf.einsum('ijk,jk->ijk', query_feature, W_q, name = 'query_feature_matmul_{}'.format(idx))
        query_feature = tf.transpose(query_feature, [0,2,1])
        
        graph = tf.add(
            tf.matmul(key_feature, query_feature),
            self.graph_bias, name='add_bias{}'.format(idx))

        graph = tf.nn.relu(graph)
        graph = tf.square(graph)
        sum_graph = tf.reduce_sum(graph, axis=2, keepdims=True)
        graph = tf.divide(graph, sum_graph)
        
        return graph


    def Predictor(self):
        """
        To using CNN on 3-dimensions embedding inputs, we have to add a new dimension on inputs
        Here the original inputs is [batch_size, num_steps, channels], we convert it into
        [batch_size, 1, num_steps, channels], hence, a conv kernel is working on a set of steps
        which has a sequential relationship with each other.
        :return:
        """
        with tf.variable_scope(self.graph_scope):

            assert self.graph_layers == 'every' if isinstance(self.graph_layers, str) else \
                len(self.graph_layers) > 0 if isinstance(self.graph_layers, list) else False, \
                'Graph layers must be \'every\' or a list of int numbers.'

            key_cnn_groups = []
            query_cnn_groups = []
            
            key_temp = self.embedding
            query_temp = self.embedding
            inputs = self.embedding

            for i in range(self.key_cnn_layers):
                key_temp = Conv1D(
                    key_temp, 128, 5,
                    1, 'SAME', 1, False, 'relu', False, 'KeyCNN_1_{}'.format(i))
                key_cnn_groups.append(key_temp)

                query_temp = Conv1D(
                    query_temp, 128, 5,
                    1, 'SAME', 1, False, 'relu', False, 'QueryCNN_1_{}'.format(i))
                query_cnn_groups.append(query_temp)
            
                graphs = self.Calculate_Graph(key_cnn_groups[i], query_cnn_groups[i], i)

                inputs = tf.matmul(graphs, inputs) # (None, 1000, 300)
                        
                inputs = tf.layers.dense(
                        inputs, self.embedding_dims, activation=tf.nn.relu,
                        use_bias=self.linear_bias, name='_{}_linear'.format(i))

        with tf.variable_scope('finalRNN_'):
            cell = tf.contrib.rnn.BasicLSTMCell(self.embedding_dims)
            predictions = []

            for i in range(self.prediction_length):

                inputs,(_, _) = tf.nn.dynamic_rnn(inputs=inputs, cell=cell, dtype=tf.float32)
                temp = tf.layers.flatten(inputs)
                if i == 0:
                    prediction = tf.layers.dense(temp, self.num_classes, reuse=False, name='_prediction_dense'+str(i))
                else:
                    prediction = tf.layers.dense(temp, self.num_classes, reuse=False, name='_prediction_dense'+str(i))
                predictions.append(prediction)

            log_predictions = tf.nn.softmax(predictions, axis=-1)
            max_log_preds = tf.reduce_max(log_predictions, axis=-1)
            self.max_log_preds = max_log_preds

    def train(self, train_generator, valid_generator, load_mode=None, load_dir=None, train_mode='graph'):
        """

        :return:
        """
        assert self.optimizer in ['sgd', 'adam'], 'Optimizer should be sgd or adam.'
        assert train_mode in ['graph', 'downstream']

        train_target = tf.negative(self.max_log_preds)
        train_var_list = self.set_trainable(train_mode)

        train_op = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(train_target, var_list=train_var_list) \
            if self.optimizer == 'sgd' else tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(train_target)
        loss = tf.reduce_mean(tf.exp(train_target) - 1)

        # TODO: Generator for data feeding.
        home = os.getcwd()
        train_generator = train_generator(os.path.join(home, "datasets", "wiki", "wiki2_train"),os.path.join(home, "data", "vocb"))
        valid_generator = valid_generator(os.path.join(home, "datasets", "wiki", "wiki2_val"),os.path.join(home, "data", "vocb"))

        with tf.Session() as self.sess:
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            if load_mode == 'last':
                self.load_checkpoints(load_dir=load_dir)
            for epoch in range(self.epochs):
                print('Starting training at Epoch {}/{}...'.format(epoch, self.epochs))
                init_loss = 0
                init_time = time.time()
                for iters in range(self.iter_per_epo):
                    train_data = next(train_generator)
                    feed_dict = {self.inputs: train_data}
                    
                    this_loss, _ = self.sess.run([loss, train_op], feed_dict=feed_dict)

                    init_loss += this_loss
                    if iters % 100 == 0 and iters != 0:
                        print(
                            'Epoch {}/{}   Iteration {}/{}   Loss={:.4f}   {:.2f}step/s'.format(
                                epoch, self.epochs, iters, self.iter_per_epo, init_loss / iters,
                                (time.time() - init_time) / iters))

                print('Epoch {} done, {} left...'.format(epoch + 1, self.epochs - epoch - 1))
                print('Training takes {:.2f} per batch...'.format((time.time() - init_time) / self.iter_per_epo))
                print('Overall loss is {:.4f} per batch...'.format(init_loss / self.iter_per_epo))

                self.saver.save(self.sess, self.ckpt_dir + 'GLoMo_{:02d}.ckpt'.format(epoch))

                if self.evaluation_per_epoch:
                    print('Start evaluation...')
                    init_loss = 0
                    init_time = 0
                    for i in range(self.evaluation_runs):
                        valid_data = next(valid_generator)
                        feed_dict = {self.inputs: valid_data}

                        this_loss = self.sess.run(loss, feed_dict=feed_dict)
                        init_loss += this_loss
                        percent = round(1.0 * i / self.evaluation_runs * 100, 2)
                        print('Evaluation : %s [%d/%d]' % (str(percent) + '%', i + 1, self.evaluation_runs), end='\r')
                    print('Evaluation done...')
                    print('Evaluation loss is {:.4f} per batch...'.format(init_loss / self.evaluation_runs))
                    print('Evaluation takes {} seconds per batch...'.format(
                        (time.time() - init_time) / self.evaluation_runs))


    def set_trainable(self, train_mode):
        """
        Training graph will allow gradients to back propagate to graph predictor and feature predictor,
        Training downstream will only allow gradients to back propagate to downstream networks.
        :return:
        """

        if train_mode == 'graph':
            tf.add_to_collection('train_var_list', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.graph_scope))
            tf.add_to_collection('train_var_list', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.feature_scope))
            return tf.get_collection('train_var_list')
        elif train_mode == 'downstream':
            return []


    def load_checkpoints(self, load_dir='./'):
        """

        :param load_mode:
        :return:
        """
        print('Loading last checkpoints...')
        indir = load_dir
        idx_list = []
        for ckpt in os.listdir(indir):
            name, ext = os.path.splitext(ckpt)
            if ext == '.ckpt':
                name_ = name.split("_")
                idx = name_[-1]
                idx_list.append(int(idx))
        idx_list.sort()
        last_ckpt_idx = idx_list[-1]
        ckpt_dir = load_dir + name_[0] + '_%s.ckpt' % str(last_ckpt_idx)
        print('Loading weights at {}...'.format(ckpt_dir))
        self.saver.restore(self.sess, ckpt_dir)
        print('Loading weights successful...')


if __name__ == '__main__':
    model = GLoMo(config) 
    model.Predictor()
    model.train(generator_train, generator_valid)