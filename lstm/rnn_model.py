import tensorflow as tf
import numpy as np

class RNN_Model(object):
    def __init__(self,config,is_training=0):
        #set the parmeters for training
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob")
        self.batch_size = tf.Variable(config.batch_size, dtype=tf.int32, trainable=False)
        num_step = config.num_step
        class_num = config.class_num
        self.input_data = tf.placeholder(tf.int32, [None, num_step])
        self.target = tf.placeholder(tf.int64, [None, class_num])
        self.mask = tf.placeholder(tf.int32, [None])
        hidden_neural_size = config.hidden_neural_size
        vocabulary_size = config.vocabulary_size
        embed_dim = config.embed_dim
        hidden_layer_num = config.hidden_layer_num
        self.new_batch_size = tf.placeholder(tf.int32, shape=[], name="new_batch_size")
        self._batch_size_update = tf.assign(self.batch_size, self.new_batch_size)
        self.pad = tf.placeholder(tf.float32, [None, 1, embed_dim, 1], name='pad')
        l2_loss = tf.constant(0.01)
        #generate word embedding for text vector
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            embedding = tf.get_variable("embedding", [vocabulary_size, embed_dim],
                                        dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        # lstm for bidirectional lstm
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(hidden_neural_size, forget_bias=1.0)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(hidden_neural_size, forget_bias=1.0)
        cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * hidden_layer_num, state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * hidden_layer_num, state_is_tuple=True)
        out_put, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                     cell_bw=cell_bw,
                                                     inputs=inputs,
                                                     dtype=tf.float32,
                                                     sequence_length=self.mask)
        out_put = tf.concat([out_put[0],out_put[1]],2)
        #sum pooling
        out_put = tf.reduce_sum(out_put,1)
        dense_dim = 1024
        #dense layer for predict
        with tf.name_scope("dense1"):
            dense_w1 = tf.get_variable("dense_w1", [hidden_neural_size*2, dense_dim], dtype=tf.float32)
            dense_b1 = tf.get_variable("dense_b1", [dense_dim], dtype=tf.float32)
            out_put = tf.matmul(out_put, dense_w1) + dense_b1
            out_put = tf.nn.tanh(out_put)
            out_put = tf.nn.dropout(out_put, self.keep_prob)
        #predict score
        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable("softmax_w", [dense_dim, class_num], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [class_num], dtype=tf.float32)
            l2_loss += tf.nn.l2_loss(softmax_w)
            l2_loss += tf.nn.l2_loss(softmax_b)
            logits = tf.matmul(out_put, softmax_w) + softmax_b
            self.logits = tf.nn.tanh(logits )
            #probability
            self.softmax = tf.nn.softmax(self.logits)

        # softmax loss
        with tf.name_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                       labels=self.target)
            self.cost = tf.reduce_mean(self.loss) + 0.01 * l2_loss

        #predict and get the accuracy
        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.softmax, axis=1)
            self.confidence = tf.reduce_max(self.softmax, axis=1)
            target = tf.argmax(self.target, axis=1)
            correct_prediction = tf.equal(self.prediction, target)
            self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        # add summary
        loss_summary = tf.summary.scalar("loss", self.cost)
        # add summary
        accuracy_summary = tf.summary.scalar("accuracy_summary", self.accuracy)
        self.out = self.softmax
        if is_training:
            return
        self.globle_step = tf.Variable(0, name="globle_step", trainable=False)
        self.lr = tf.Variable(0.0, trainable=False)
        #add summary
        self.summary = tf.summary.merge([loss_summary, accuracy_summary])
        #SGD optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost)
        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self.new_lr)

    #update learning rate
    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value})
    # update data batch size
    def assign_new_batch_size(self,session,batch_size_value):
        session.run(self._batch_size_update,feed_dict={self.new_batch_size:batch_size_value})
