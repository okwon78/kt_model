import argparse
import os
import sys
import logging

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None
IMAGE_SIZE = 784


def import_data():
    return input_data.read_data_sets(FLAGS.data_dir, fake_data=FLAGS.fake_data)


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    return tf.get_variable(name='weights', initializer=tf.truncated_normal(shape=shape))


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    return tf.get_variable(name='bias', initializer=tf.constant(0.1, shape=shape))


def variable_summaries(name, tensors):
    """Attach a log of summaries to a Tensor (for TensorBoard visualization)."""
    mean = tf.reduce_mean(tensors)
    tf.summary.scalar(name + '_mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(tensors - mean)))
    tf.summary.scalar(name + '_stddev', stddev)
    tf.summary.scalar(name + '_max', tf.reduce_max(tensors))
    tf.summary.scalar(name + '_min', tf.reduce_min(tensors))
    tf.summary.histogram(name + '_histogram', tensors)


def nn_layer(name, input_tensor, input_dim, output_dim, act, dropout_keep_prob=None):
    with tf.variable_scope(name):
        weights = weight_variable(shape=(input_dim, output_dim))
        variable_summaries('weights', weights)
        biases = bias_variable(shape=(output_dim,))
        variable_summaries('biases', biases)
        logits = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('logits', logits)
        activations = act(logits, name='activation')
        tf.summary.histogram('activations', activations)

        if dropout_keep_prob is not None:
            tf.summary.scalar('dropout_keep_probability', dropout_keep_prob)
            return tf.nn.dropout(activations, dropout_keep_prob)
        else:
            return activations


def train(data):
    with tf.name_scope('input'):
        x = tf.placeholder(dtype=tf.float32, shape=(None, IMAGE_SIZE), name='x-input')
        y_ = tf.placeholder(dtype=tf.int64, shape=(None,), name='y-input')
        image_shaped_input = tf.reshape(x, shape=(-1, 28, 28, 1))
        tf.summary.image(name='images', tensor=image_shaped_input)
        keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

    hidden1 = nn_layer(name='first',
                       input_tensor=x,
                       input_dim=IMAGE_SIZE, output_dim=500,
                       act=tf.nn.relu,
                       dropout_keep_prob=None)

    hidden2 = nn_layer(name='second',
                       input_tensor=hidden1,
                       input_dim=500, output_dim=300,
                       act=tf.nn.relu,
                       dropout_keep_prob=keep_prob)

    y = nn_layer(name='last',
                 input_tensor=hidden2,
                 input_dim=300, output_dim=10,
                 act=tf.identity,
                 dropout_keep_prob=None)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    def feed_dict(train):

        if train or FLAGS.fake_data:
            xs, ys = data.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = data.test.images, data.test.labels
            k = 1.0

        return dict({
            x: xs,
            y_: ys,
            keep_prob: k
        })

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        sess.run(tf.global_variables_initializer())
        for i in range(FLAGS.max_steps):
            if i % 10 == 0:
                summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
                test_writer.add_summary(summary, i)
            else:
                if i % 100 == 99:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
                                          feed_dict=feed_dict(True),
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)
                else:
                    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                    train_writer.add_summary(summary, i)

        train_writer.close()
        test_writer.close()

    for name in [n.name for n in tf.get_default_graph().as_graph_def().node]:
        print(name)


def main(_):
    logging.basicConfig(filename='kt.log', level=logging.ERROR)

    if tf.gfile.Exists(FLAGS.log_dir):
        print('Delete previous log dir')
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # if not tf.gfile.Exists(FLAGS.data_dir):
    #     tf.gfile.MakeDirs(FLAGS.data_dir)

    data = import_data()
    train(data)


if __name__ == '__main__':
    print('start running')

    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data',
                        type=bool,
                        nargs='?', const=True, default=False,
                        help='If true, it uses fake data for unit test')
    parser.add_argument('--max_steps',
                        type=int,
                        nargs='?', const=100, default=1000, help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--dropout',
                        type=float, default=0.9, help='Keep probability for training dropout.')
    parser.add_argument('--data_dir',
                        type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'),
                        help='Directory for storing input data')
    parser.add_argument('--log_dir',
                        type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log'),
                        help='Summaries log directory')

    FLAGS, unparsed = parser.parse_known_args()

    print('fake_data: ', FLAGS.fake_data)
    print('max_steps: ', FLAGS.max_steps)
    print('learning_rate: ', FLAGS.learning_rate)
    print('dropout: ', FLAGS.dropout)
    print('data_dir: ', FLAGS.data_dir)
    print('log_dir: ', FLAGS.log_dir)

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
