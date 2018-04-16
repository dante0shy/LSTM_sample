import tensorflow as tf
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size',64,'the batch_size of the training procedure')
flags.DEFINE_float('lr',0.01,'the learning rate')
flags.DEFINE_float('lr_decay',0.89,'the learning rate decay')
flags.DEFINE_integer('vocabulary_size',4000,'vocabulary_size')
flags.DEFINE_integer('emdedding_dim',128,'embedding dim')
flags.DEFINE_integer('hidden_neural_size',128,'LSTM hidden neural size')
flags.DEFINE_integer('hidden_layer_num',1,'LSTM hidden layer num')
flags.DEFINE_string('dataset_path','word_data.pkl','dataset path')
flags.DEFINE_integer('max_len',200,'max_len of training sentence')
flags.DEFINE_integer('valid_num',64,'epoch num of validation')
flags.DEFINE_integer('checkpoint_num',15,'epoch num of checkpoint')
flags.DEFINE_float('init_scale',0.1,'init scale')
flags.DEFINE_integer('class_num',2,'class num')
flags.DEFINE_float('keep_prob',0.5,'dropout rate')
flags.DEFINE_integer('num_iter',5000,'num epoch')
flags.DEFINE_integer('max_decay_epoch',50,'num epoch')
flags.DEFINE_integer('max_grad_norm',5,'max_grad_norm')
flags.DEFINE_string('out_dir',os.path.abspath(os.path.join(os.path.curdir,"model")),'output directory')
flags.DEFINE_integer('check_point_every',15,'checkpoint every num epoch ')
flags.DEFINE_integer('max_vector_len',200,'max vector len ')

class Config(object):
    hidden_neural_size = FLAGS.hidden_neural_size
    vocabulary_size = FLAGS.vocabulary_size
    embed_dim = FLAGS.emdedding_dim
    hidden_layer_num = FLAGS.hidden_layer_num
    class_num = FLAGS.class_num
    keep_prob = FLAGS.keep_prob
    lr = FLAGS.lr
    lr_decay = FLAGS.lr_decay
    batch_size = FLAGS.batch_size
    num_step = FLAGS.max_len
    max_grad_norm = FLAGS.max_grad_norm
    num_iter = FLAGS.num_iter
    max_decay_epoch = FLAGS.max_decay_epoch
    valid_num = FLAGS.valid_num
    out_dir = FLAGS.out_dir
    checkpoint_every = FLAGS.check_point_every
    max_vector_len = FLAGS.max_vector_len
    init_scale = FLAGS.init_scale