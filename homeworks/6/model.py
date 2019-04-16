import tensorflow as tf
import yaml
import os
from image_dataset import ImageDataset
from utils import AttrDict
from tensorflow.contrib.slim.nets import resnet_v2
from pprint import pprint
from glob import glob

class TransferModel(object):
    
    #This is the meat of the model and where most of the fun stuff happens.

    def __init__(self, sess, flags):
        #load self.config
        self.sess = sess
        stream = open(flags.config_yaml, 'r')
        config = yaml.load(stream)
        self.config = AttrDict(config)

        self.config_d = config
        self.model_id = flags.model_id
        self.data_dir = flags.data_dir

        self.set_model_params(self.config)
        if not flags.train:
            self.test_dataset = ##YOUR CODE HERE, INITIALIZE APPROPRIATE DATASET OBJECT
        self.path = 'pretrained_models/resnet_v2_50.ckpt'
        self.build_graph()

    def set_model_params(self, config):
        #This function is finished
        self.num_classes = config.num_classes
        self.height = config.height
        self.width = config.width
        self.write_images = config.write_images
        self.batch_size = config.batch_size
        

    def net(self):
        #This function is finished.
        #Try to figure out what endpoints are and why they might be useful.
        #This creates your model architecture, a 50 layer resnet.
        net, end_points = resnet_v2.resnet_v2_50(
            self.inputs,
            num_classes=self.num_classes,
            is_training=self.is_train,
            global_pool=True, #check out this global pooling ;) 
            output_stride=None,
            reuse=tf.AUTO_REUSE,
            scope="resnet_v2_50"
        )
        return net, end_points
            
    def build_graph(self):
        # Make some placeholders first: what do you need to pass into the graph in order to get out what you want?
        # Hints: there are three necessary for training
        self.y_hat = #YOUR CODE HERE (multiple lines)
        self.probabilities = tf.nn.softmax(self.y_hat) #Do not do softmax cross entropy on this
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.y_hat, labels=self.labels), name='sftmxce_loss')
        self.accuracy = #??

        loss_sum = tf.summary.scalar('loss', self.loss)
        acc_sum = tf.summary.scalar('accuracy', self.accuracy)
        if self.write_images:
            img_sum = tf.summary.image('train_images', ... )
        img_summary = None
        self.summary = tf.summary.merge_all()
        self.opt = #??
        self.saver = tf.train.Saver(max_to_keep=10)

    def make_train_graph(self):
        vars_to_train = #?? Start with tf.trainable_variables

        if self.config.weight_decay > 0:
            weight_norm = #??
            self.loss += self.config.weight_decay * weight_norm
        pprint(vars_to_train)
        
        self.train_step = self.opt.minimize(self.loss, var_list=vars_to_train)

        if self.model_id != -1:
            start_epoch = self.restore()
        else:
            self.init_new_model()
            start_epoch = 0

        log_dir = os.path.join(self.model_dir, 'logs/train')
        val_dir = os.path.join(self.model_dir, 'logs/val')
        
        self.train_writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.val_writer = tf.summary.FileWriter(val_dir, self.sess.graph)

        self.train_dataset = ImageDataset(
            data_dir=os.path.join(self.data_dir, 'train'),
            h=self.height,
            w=self.width,
            batch_size=self.batch_size,
            crop_proportion=0.8
        )
        self.val_dataset = ImageDataset(
            #what's the difference
        )
        return start_epoch
        

    def train(self):
        start_epoch = self.make_train_graph()
        idx = 0
        print("Starting training")
        #YOUR CODE HERE: Implement training loop:
        for epoch in range(start_epoch, self.config.num_epochs):
            print("idx {} loss {} acc {}".format(idx, batch_loss, batch_acc))
            print("VAL idx {} loss {} acc {}".format(idx, v_batch_loss, v_batch_acc))
        print("Done Training")

    def predict(self, path):
        input_ = #?? feel free to create a dataset object ;) 
        feed_dict={self.inputs:input_, self.is_train:False}
        probs = self.sess.run(self.probabilities, feed_dict=feed_dict)
        probs = probs[0]
        print(probs)

    def init_new_model(self):
        #This funny function is done for ya
        assert self.model_id < 0
        try:
            self.model_id = 1 + max([
                int(f.split('/')[-1][1:]) for f in glob('experiments/m*')
            ])
        except ValueError:
            self.model_id = 0

        self.model_dir = os.path.join('./experiments', 'm'+str(self.model_id))
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        with open(os.path.join(self.model_dir, 'config.yaml'), 'w') as out:
            yaml.dump(self.config_d, out, default_flow_style=False)
        start_epoch = 0
        tvs = tf.trainable_variables()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
    
        vars_to_load = [var for var in tvs if 'biases' not in var.name and 'logits' not in var.name]
        saver = tf.train.Saver(vars_to_load)
        saver.restore(self.sess, self.path)

    def restore(self):
        #This is also done for you u lucky ducks
        assert self.model_id >= 0, "Trying to restore a negative model id"
        self.model_dir = os.path.join('./experiments', 'm{}'.format(self.model_id))
        stream = open(os.path.join(self.model_dir, 'config.yaml'))
        loaded_train_config = yaml.load(stream)
        pprint(loaded_train_config)
        self.sess.run(tf.global_variables_initializer())
        checkpoint_dir = os.path.join(self.model_dir, 'ckpts')
        print("Checkpoint dir is {}".format(checkpoint_dir))
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        print("checkpoint path is {}".format(checkpoint_path))
        start_epoch = 1 + int(checkpoint_path.split('.ckpt')[1].split('-')[1])
        print("Restoring from {} with epoch {}".format(checkpoint_path, start_epoch))
        self.saver.restore(self.sess, checkpoint_path)
        return start_epoch

    def save(self, idx, epoch):
        #This too, but get how it works
        self.saver.save(
            self.sess,
            os.path.join(self.model_dir, 'ckpts/model.ckpt'),
            global_step=epoch,
            write_meta_graph=not bool(epoch)
        )
