import scipy.misc
import os
import numpy as np
from activation_functions import Heaviside


class ImageProc:
    def __init__(self, batch_size, directory):
        self.directory = directory
        self.batch_size = batch_size
        self.filelist = os.listdir(self.directory)
        self.pictures = []


    def get_single_image(filename):
        png = scipy.misc.imread(filename) 
        vect_png = np.asarray(png).reshape(-1)
        vect_proc_png = ImageProc.transform(vect_png)
        return vect_proc_png

    
    def get_pictures(self):
        png = None
        filebatch = self.filelist[:self.batch_size]
        for filename in filebatch:
            png = scipy.misc.imread(self.directory + filename)
            vect_png = np.asarray(png).reshape(-1)
            vect_proc_png = ImageProc.transform(vect_png)
            self.pictures.append(vect_proc_png)
        return self.pictures
            

    def transform(png):
        vf = np.vectorize(Heaviside)
        return vf(png)