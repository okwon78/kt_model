import tensorflow as tf
import argparse
import os
import sys

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train():
    pass


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ is '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data',
                        type=bool,
                        nargs='?', const=True, default=False,
                        help='If true, it uses fake data for unit test')
    parser.add_argument('--max_steps',
                        type=int,
                        nargs='?', const=100, default=1000, help='Number of steps to run trainer.')
    parser.add_arguemtn('--learning_rate',
                        type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--dropout',
                        type=float, default=0.9, help='Keep probability for training dropout.')
    parser.add_argument('--data_dir',
                        type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '/data'),
                        help='Directory for storing input data')
    parser.add_argument('--log_dir',
                        type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '/log'),
                        help='Summaries log directory')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
