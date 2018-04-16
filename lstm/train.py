from data_process import data_loader
import numpy as np
import os
import time
from lstm.rnn_model import RNN_Model
from lstm.tf_config import Config
import tensorflow as tf




def evaluate(sess,val_model, data, global_step=None,val_summary = None):
    correct_num = 0
    total_num = 0
    for step, (x, mask_x, y) in enumerate(data.get_data(batch_size=Config.batch_size)):

        # fetches = val_model.accuracy
        fetches = [val_model.correct_num,val_model.prediction]
        feed_dict = {}
        feed_dict[val_model.input_data] = x
        feed_dict[val_model.target] = y
        feed_dict[val_model.mask] = mask_x
        feed_dict[val_model.keep_prob] = 1.
        val_model.assign_new_batch_size(sess, len(x))
        feed_dict[val_model.pad] = np.zeros([Config.batch_size, 1, Config.embed_dim, 1])
        count,pre = sess.run(fetches, feed_dict)
        total_num += len(x)
        # correct_num = correct_num + (count-correct_num)*len(x)/total_num
        correct_num += count
    accuracy = float(correct_num)/total_num
    dev_summary = tf.summary.scalar('dev_accuracy', accuracy)
    dev_summary = sess.run(dev_summary)
    if val_summary:
        val_summary.add_summary(dev_summary, global_step)
        val_summary.flush()
    return accuracy

def run_epoch(sess,model,val_model,data,data_val,global_step,train_summary,val_summary):
    num = 0
    accuracy_t = 0
    cost_t = 0
    for step, (x,mask,y) in enumerate(data.get_data(batch_size=Config.batch_size)):

        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.target] = y
        feed_dict[model.mask] = mask
        feed_dict[model.keep_prob] = Config.keep_prob
        feed_dict[model.pad] = np.zeros([Config.batch_size, 1, Config.embed_dim, 1])

        # sess.run(model.optimizer, feed_dict)
        model.assign_new_batch_size(sess, len(x))
        fetches = [model.optimizer,model.cost, model.accuracy, model.summary,model.out]

        optimizer,cost, accuracy, summary,out = sess.run(fetches, feed_dict)
        # print out

        train_summary.add_summary(summary, global_step)
        train_summary.flush()
        accuracy_t += len(x)*accuracy
        cost_t += len(x)*(cost-cost_t)/(len(x)+num)
        num+=len(x)
        # if (global_step % 100 == 0):

        global_step += 1
        # if (global_step >5):
        #     break
    valid_accuracy = evaluate(sess, val_model, data_val, global_step, val_summary)
    print("the %i step, train cost is: %f and the train accuracy is %f and the valid accuracy is %f" % (
        global_step, cost, accuracy, valid_accuracy))
    print("the %i step, train cost_T is: %f and the train accuracy_T is %f " % (
        global_step, cost_t, accuracy_t/num))
    return  global_step

def train_step():
    print 'load date'
    config = Config()
    eval_config = Config()
    eval_config.keep_prob = 1.0

    loader_train = data_loader.Data_loader(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_for_train/train.json"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_for_train/word_list.json"),max_len=config.max_vector_len)
    loader_val = data_loader.Data_loader(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_for_train/val.json"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_for_train/word_list.json"),max_len=config.max_vector_len)
    with tf.Graph().as_default(),tf.Session() as sess:
        initializer = tf.truncated_normal_initializer(stddev=config.init_scale)
        with tf.variable_scope("model",initializer=initializer):
            model = RNN_Model(config=config)

        with tf.variable_scope("model",reuse=True):#,initializer=initializer
            valid_model = RNN_Model(config=eval_config,is_training=1)

        train_summary_dir = os.path.join(config.out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # dev_summary_op = tf.merge_summary([valid_model.loss_summary,valid_model.accuracy])
        val_summary_dir = os.path.join(eval_config.out_dir, "summaries", "val")
        val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints_v1"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        ckpt = None
        ckpt = tf.train.get_checkpoint_state(os.path.join(config.out_dir, "checkpoints_v1"))
        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training from the model {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        # tf.global_variables_initializer().run()
        else:
            tf.initialize_all_variables().run()

        global_steps = 1
        begin_time = int(time.time())

        for i in range(config.num_iter):
            print("the %d epoch training..."%(i+1))
            lr_decay = config.lr_decay ** max(i-config.max_decay_epoch,0.0)
            model.assign_new_lr(sess,config.lr*lr_decay)
            global_steps=run_epoch(sess, model, valid_model, loader_train,loader_val, global_steps, train_summary_writer, val_summary_writer)

            if i% config.checkpoint_every==0:
                path = saver.save(sess,checkpoint_prefix,global_steps)
                print("Saved model chechpoint to{}\n".format(path))
                # break
            # break
        print("the train is finished")
        end_time=int(time.time())
        print("training takes %d seconds already\n"%(end_time-begin_time))
        test_accuracy=evaluate(valid_model, sess, loader_val)
        print("the test data accuracy is %f"%test_accuracy)
        print("program end!")



def main(_):
    train_step()


if __name__ == "__main__":
    tf.app.run()
