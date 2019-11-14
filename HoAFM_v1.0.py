import logging
import sklearn
import numpy as np
import tensorflow as tf
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import log_loss
import math
from sklearn.metrics import mean_squared_error, auc, roc_curve
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import config
import DataLoader_x
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='log/train_xNFM_{0}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S')), level=logging.DEBUG,
                    format=LOG_FORMAT)


def log_info(message):
    logging.info(time.asctime(time.localtime(time.time())) + message)
    print(message)


dfm_params = {
    "dataset": 'criteo',
    "feature_size": 2088202,
    "field_size": 39,
    "embedding_size": 64,
    "activation": tf.nn.leaky_relu,
    "epoch": 10,
    "deep_layers": [64],
    "batch_size": 1024,
    "learning_rate": 0.02,
    "keep": 0.3,
    "deep_keep": [0.3],
    "xn_activation": tf.identity,  # tf.nn.leak_relu
    "xn_keep": [1.0],
    "optimizer": "adagrad",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.0,
    "verbose": True,
    "random_seed": 2019,
    "early_stop": 1,
    "loss_type": 'log_loss',
    "pretrain_flag": -1,
    "save_file": '',
}


class GFM(BaseEstimator, TransformerMixin):
    def __init__(self,
                 dataset='criteo',
                 feature_size=2088202,
                 field_size=39,
                 embedding_size=64,
                 activation=tf.nn.leaky_relu,
                 epoch=10,
                 deep_layers=[64],
                 batch_size=1024,
                 learning_rate=0.01,
                 keep=1,
                 deep_keep=0.5,
                 xn_activation=tf.identity,  # tf.nn.leak_relu,
                 xn_keep=[1.0],
                 optimizer="adagrad",
                 batch_norm=0,
                 batch_norm_decay=0.995,
                 l2_reg=0.0,
                 verbose=False,
                 random_seed=2019,
                 loss_type="log_loss",
                 early_stop=1,
                 pretrain_flag=-1,
                 save_file='',
                 ):
        assert loss_type in ["log_loss", "mse"], \
            "loss_type can be either 'log_loss' for classification task or 'mse' for regression task"

        self.dataset = dataset
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.activation = activation
        self.epoch = epoch
        self.deep_layers = deep_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep = keep
        self.deep_keep = deep_keep
        self.xn_activation = xn_activation
        self.xn_keep = xn_keep
        self.optimizer_type = optimizer
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.l2_reg = l2_reg

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.early_stop = early_stop
        self.pretrain_flag = pretrain_flag
        self.save_file = save_file
        # performance of each epoch
        self.train_logloss, self.valid_logloss, self.test_logloss = [], [], []

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.train_Xi = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            # self.train_Xv = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1], name='train_labels')
            self.dropout_keep = tf.placeholder(tf.float32)
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_Xi)  # N * F * K
            # first order term
            self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_Xi)
            self.y_first_order = tf.reduce_sum(self.y_first_order, 1)
            self.Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out = tf.add_n((self.y_first_order, self.Bias))

            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size, self.embedding_size])
            for i in range(0, len(self.deep_layers)):
                A_deep_order = self.embeddings
                slice_list = []
                for j in range(self.field_size):
                    slice_left = self.y_deep[:, 0:j, :]
                    slice_right = self.y_deep[:, j + 1:, :]
                    slice_j = tf.concat((slice_left, slice_right), axis=1)
                    # self.weights['field_weight_layer%d_slice%d' % (i, j)] = tf.contrib.sparsemax.sparsemax(
                    #    self.weights['field_weight_layer%d_slice%d' % (i, j)])
                    self.weights['field_weight_layer%d_slice%d' % (i, j)] = tf.math.l2_normalize(
                        self.weights['field_weight_layer%d_slice%d' % (i, j)], axis=0)

                    y_slice_j = tf.multiply(slice_j, self.weights['field_weight_layer%d_slice%d' % (i, j)])
                    slice_list.append(tf.reduce_sum(y_slice_j, axis=1))
                B_deep_order = tf.transpose(slice_list, perm=[1, 0, 2])
                # B_deep_order = self.xn_activation(B_deep_order)
                # B_deep_order = tf.nn.dropout(B_deep_order, self.xn_keep[i])
                self.y_deep = tf.multiply(A_deep_order, B_deep_order)
                # self.y_deep = self.activation(self.y_deep)
                # self.y_deep = tf.nn.dropout(self.y_deep, self.xn_keep[i])

                # the i-th order output

                self.y_deep_comp = tf.reduce_sum(self.y_deep, axis=1)
                if self.batch_norm:
                    self.y_deep_comp = self.batch_norm_layer(self.y_deep_comp, train_phase=self.train_phase,
                                                             scope_bn="y_deep_comp%d" % i)
                # self.y_deep_comp = self.activation(self.y_deep_comp)
                self.y_deep_comp = tf.nn.dropout(self.y_deep_comp, self.dropout_keep)
                self.y_deep_comp = tf.matmul(self.y_deep_comp, self.weights['comp_weight_%d' % i])  # None * 1
                self.out = tf.add(self.out, self.y_deep_comp)

                self.y_deep = self.activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.deep_keep[i])

            if self.loss_type == "log_loss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.reduce_mean(tf.losses.log_loss(predictions=tf.reshape(self.out, [-1]),
                                                              labels=tf.reshape(self.train_labels, [-1])))
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))

            # l2 regularization on weights
            if self.l2_reg > 0:
                emb_loss = tf.nn.l2_loss(self.embeddings) * self.embed_l2
                self.loss += emb_loss

            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                log_info("#params: %d" % total_parameters)

    def _initialize_weights(self):
        weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        if self.pretrain_flag > 0:  # with pretrain
            pretrain_file = 'pretrain/%s_%d/%s_%d' % (
                args['dataset'], args['embedding_size'], args['dataset'], args['embedding_size'])
            weight_saver = tf.train.import_meta_graph(pretrain_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, pretrain_file)
                fe, fb, b = sess.run([feature_embeddings, feature_bias, bias])
            weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32)
            weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32)
            weights['bias'] = tf.Variable(b, dtype=tf.float32)
        else:  # without pretrain
            # embeddings
            weights['feature_embeddings'] = tf.Variable(initializer([self.feature_size, self.embedding_size]),
                                                        name='feature_embeddings')
            weights['feature_bias'] = tf.Variable(initializer([self.feature_size, 1]), name='feature_bias')
            weights['bias'] = tf.Variable(initializer([1]), name='bias')
            # weights['bias'] = tf.Variable(tf.constant(0.01), name='bias')  # 1 * 1

        # deep layers
        for i in range(0, len(self.deep_layers)):
            weights["A_w_%d" % i] = tf.Variable(initializer([self.embedding_size, self.deep_layers[i]]),
                                                name="A_w_%d" % i)
            if i == 0:
                pre_emb_size = self.embedding_size
            else:
                pre_emb_size = self.deep_layers[i - 1]
            weights["B_w_%d" % i] = tf.Variable(initializer([pre_emb_size, self.deep_layers[i]]),
                                                name="B_w_%d" % i)
            weights['comp_weight_%d' % i] = tf.Variable(initializer([self.deep_layers[i], 1]),
                                                        name='comp_weight_%d' % i)
            weights['comp_bias_%d' % i] = tf.Variable(initializer([1]), name='comp_bias_%d' % i)

            for j in range(0, self.field_size):
                weights['field_weight_layer%d_slice%d' % (i, j)] = tf.Variable(
                    initializer([self.field_size - 1, self.deep_layers[i]]), name='field_weight_layer%d_slice%d' % (i, j))

        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_Xi: data['Xi'],
                     self.train_labels: data['Y'],
                     self.dropout_keep: self.keep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        Xi, Y = [], []
        # forward get sample
        i = start_index
        while len(Xi) < batch_size and i < len(data['Xi']):
            if len(data['Xi'][i]) == len(data['Xi'][start_index]):
                Y.append(data['Y'][i])
                Xi.append(data['Xi'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(Xi) < batch_size and i >= 0:
            if len(data['Xi'][i]) == len(data['Xi'][start_index]):
                Y.append(data['Y'][i])
                Xi.append(data['Xi'][i])
                i = i - 1
            else:
                break
        return {'Xi': Xi, 'Y': Y}

    def shuffle_in_unison_scary(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def train(self, Train_data, Validation_data, Test_data):  # fit a dataset
        # Check Init performance
        if self.verbose > 0:
            t2 = time.time()
            t2 = time.time()
            init_train, auc_train = self.evaluate(Train_data)
            init_valid, auc_valid = self.evaluate(Validation_data)
            init_test, auc_test = self.evaluate(Test_data)
            log_info("Init logloss: \t train=%.4f, validation=%.4f, test=%.4f [%.1f s]" % (
                init_train, init_valid, init_test, time.time() - t2))
            log_info("Init auc: \t train=%.4f, validation=%.4f, test=%.4f" % (auc_train, auc_valid, auc_test))

        for epoch in range(self.epoch):
            t1 = time.time()
            self.shuffle_in_unison_scary(Train_data['Xi'], Train_data['Y'])
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                self.partial_fit(batch_xs)

                # print('weight:', self.weights['field_weight_layer0_slice0'].eval(session=self.sess))

            t2 = time.time()

            # output validation
            train_result, train_auc = self.evaluate(Train_data)
            valid_result, valid_auc = self.evaluate(Validation_data)
            test_result, test_auc = self.evaluate(Test_data)
            self.train_logloss.append(train_result)
            self.valid_logloss.append(valid_result)
            self.test_logloss.append(test_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                log_info("Epoch %d [%.1f s]\ttrain=%.4f, validation=%.4f, test=%.4f [%.1f s]"
                         % (epoch + 1, t2 - t1, train_result, valid_result, test_result, time.time() - t2))
                log_info("auc: \t train=%.4f, validation=%.4f, test=%.4f" % (train_auc, valid_auc, test_auc))
            if self.early_stop > 0 and self.eva_termination(self.valid_logloss):
                # log_info "Early stop at %d based on validation result." %(epoch+1)
                break

    def eva_termination(self, valid):
        if self.loss_type == 'square_loss':
            if len(valid) > 2:
                if valid[-1] > valid[-2]:
                    return True
        else:
            if len(valid) > 2:
                if valid[-1] > valid[-2]:
                    return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        predictions = None
        iter_num = np.ceil(num_example / config.MAX_EVALUTION_NUM).astype(int)
        for num in range(iter_num):
            start_num = num * config.MAX_EVALUTION_NUM
            end_num = (num + 1) * config.MAX_EVALUTION_NUM
            if end_num > num_example:
                end_num = num_example
            feed_dict = {self.train_Xi: data['Xi'][start_num: end_num],
                         self.train_labels: data['Y'][start_num: end_num],
                         self.dropout_keep: 1.0,
                         self.train_phase: True}
            if predictions is None:
                predictions = self.sess.run((self.out), feed_dict=feed_dict)
            else:
                predictions = np.vstack((predictions, self.sess.run((self.out), feed_dict=feed_dict)))
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(data['Y'], (num_example,))
        if self.loss_type == 'square_loss':
            predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
            predictions_bounded = np.minimum(predictions_bounded,
                                             np.ones(num_example) * max(y_true))  # bound the higher values
            RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
            return RMSE
        elif self.loss_type == 'log_loss':
            logloss = log_loss(y_true, y_pred, eps=1e-7)

            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            auc = sklearn.metrics.auc(fpr, tpr)
            return logloss, auc


def eval_model_params(model, start_time):
    # Find the best validation result across iterations
    best_valid_score = 0
    if dfm_params['loss_type'] == 'square_loss':
        best_valid_score = min(model.valid_logloss)
    elif dfm_params['loss_type'] == 'log_loss':
        best_valid_score = min(model.valid_logloss)
    best_epoch = model.valid_logloss.index(best_valid_score)
    log_info("Best Iter(validation)= %d\t train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
             % (best_epoch + 1, model.train_logloss[best_epoch], model.valid_logloss[best_epoch],
                model.test_logloss[best_epoch],
                time.time() - start_time))


if __name__ == '__main__':
    args = dfm_params
    feat_dim, dict_train, dict_val, dict_test = DataLoader_x.load_Xi_Xv_Y('data/criteo/criteo/', '', 2088202)
    #args['dataset'] = 'avazu'
    #args['feature_size'] = 1544294
    #args['field_size'] = 22
    #args['embedding_size'] = 10
    #args['keep'] = 0.8
    #feat_dim, dict_train, dict_val, dict_test = DataLoader_x.load_Xi_Xv_Y('avazu/T4/', '', 1544294)

    #args['deep_keep'] = [0.2]
    #args['deep_layers'] = [10]

    args['learning_rate'] = 0.01
    keep_arr = [0.8]
    xn_keep_arr = [[1.0]]

    for i in range(len(xn_keep_arr)):
        for j in range(len(keep_arr)):
            args['xn_keep'] = xn_keep_arr[i]
            args['keep'] = keep_arr[j]
            param_str = ''
            for key, item in args.items():
                param_str += '{} : {}, '.format(key, item)
            log_info(param_str)
            start_time = time.time()
            log_info('start to train model. ')
            dfm = GFM(**args)
            dfm.train(dict_train, dict_val, dict_test)
            log_info('training model done. ')
            eval_model_params(dfm, start_time)
            log_info(param_str)
