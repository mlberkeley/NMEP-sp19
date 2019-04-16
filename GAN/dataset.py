import tensorflow as tf
from glob import glob
import os

class Dataset(object):
    '''
    https://www.tensorflow.org/guide/datasets
    https://www.tensorflow.org/guide/performance/datasets
    Dataset object which returns appropriate datasets for class A and class B.
    Alternatively, you can have this object only get a single dataset, ie one for
    just one class and have multiple. this is a design choice left up to you, but this one
    assumes you do it all 
    '''
    
    def __init__(self, args):
        A_paths = './data/{}/A/*.jpg'.format(args.type) #args.type is train or test
        B_paths = './data/{}/B/*.jpg'.format(args.type) 
        self.A_filenames = glob(A_paths)
        self.B_filenames = glob(B_paths)
        self.tf_A_filenames = tf.constant(self.A_filenames)
        self.tf_B_filenames = tf.constant(self.B_filenames)

    def _parse_function(self, filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        return image_decoded

    def _augment_img(self, image):
        #Must be implemented entirely with tensorflow transformations. 
        pass

    def _get_dataset_from_files(self, tf_files):    
        dataset = tf.data.from_tensor_slices(tf_files)
        dataset = dataset.map(
            lambda filename: self._parse_function(filename),
            num_parallel_calls = self.num_parallel_calls
        )
        dataset = dataset.map(
            lambda img: self.augment_img(img),
            num_parallel_calls = self.num_parallel_calls
        )
        dataset = dataset.batch(batch_size = self.batch_size)
        dataset = dataset.prefetch(buffer_size = self.prefetch_buffer_size)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat()
        return dataset
        
    def get_datasets(self):
        pass
        
    
