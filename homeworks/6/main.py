import os
import tensorflow as tf
from model import TransferModel
import yaml

FLAGS = tf.app.flags
FLAGS.DEFINE_boolean("train", False, "whether to train or infer")
FLAGS.DEFINE_string("data_dir", "./data/", "directory where data is stored")
FLAGS.DEFINE_string("config_yaml", "./config.yaml", "hyperparameters for training")
FLAGS.DEFINE_integer("model_id", -1, "If positive: restores the model given by the id")
FLAGS.DEFINE_string("image_path", None, "If not none: runs prediction model on this image")

FLAGS=FLAGS.FLAGS

def main(_):
    with tf.Session() as sess:
        model = TransferModel(sess, FLAGS)

        if FLAGS.train:
            model.train()
        else:
            if FLAGS.image_path is not None:
                model.restore()
                model.predict(FLAGS.image_path)
            #YOUR CODE HERE
            #Add logic, flags, etc. to test on a directory of images.

if __name__ == '__main__':
    tf.app.run()

