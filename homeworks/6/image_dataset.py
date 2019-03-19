import numpy as np
import random
import os
from glob import glob
from scipy import misc
from skimage.transform import resize

class ImageDataset(object):
    ###Image dataset using skimage/numpy, slower than tf data api

    def __init__(self, data_dir, h, w, batch_size, crop_proportion, glob_pattern='*/*.jpg'):
        self.data_dir = data_dir
        self.h = h
        self.w = w
        self.batch_size = batch_size
        self.idx = 0
        self.crop_proportion = crop_proportion
        self.MEAN = np.reshape(np.array([0.485, 0.458, .407]), [1,1,3])
        #Keep in mind we don't augment val data
        if glob_pattern == None:
            self.do_y = False
            self.filenames = [self.data_dir]
            self.N = 1
            return

        self.get_label_dict()
        self.do_y = True
        self.filenames = glob(os.path.join(self.data_dir, glob_pattern))
        if 'val' in data_dir:
            self.filenames = self.filenames[:140]
        self.N = len(self.filenames)
        print(self.N)
        self.num_labels=len(self.label_list)
        self.labels = [self.label_dict[f.split('/')[-2]] for f in self.filenames]

    def get_label_dict(self):
        #Maps from labels to numbers
        dirs = glob(os.path.join(self.data_dir, '*'))
        self.label_dict = {}
        self.label_list = [d.split('/')[-1] for d in dirs]
        self.label_list.sort()
        for i, d in enumerate(self.label_list):
            self.label_dict[d] = i

    def new_epoch(self):
        x_y = list(zip(self.filenames, self.labels))
        random.shuffle(x_y)
        self.filenames, self.labels = zip(*x_y)
        self.idx = 0

    def get_next_batch(self):
        #TODO: YOUR CODE HERE
        #Should load filenames, augment etc.

        return batch, batch_y

    def load_image(self, filename):
        image = misc.imread(filename, mode='RGB')
        #Load and square center crop your image:
        #YOUR CODE HERE: CENTER CROP THE IMAGE
        resized = resize(crop, (self.load_h, self.load_w))
        return resized

    def random_crop(self, image):
        #Implement Random Cropping
        return cropped

    def augment_image(self, image):
        image = image - self.MEAN
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        if self.crop_proportion is not None:
            image = self.random_crop(image)
        return image

        #BONUS: More augmentation
        
