import os
import random
import numpy as np
import pandas as pd
from time import time, sleep
from collections import Counter
from multiprocessing import Pool
from matplotlib import pyplot as plt

from cv2 import imread as cv2_imread
from cv2 import resize as cv2_resize
from cv2 import INTER_AREA, INTER_LINEAR, INTER_NEAREST
from cv2 import cvtColor
from cv2 import COLOR_BGR2RGB, COLOR_RGB2HSV, COLOR_HSV2RGB

data_dir = '/media/64GB/datasets/github_avatars/imgs_64'

def load_path(path):
    img = cv2_imread(path)
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
    img = cvtColor(img, COLOR_BGR2RGB)
    return img

def load(ntrain=2000, ntest=1000):
	pool = Pool(8)
	fs = [os.path.join(data_dir, path) for path in os.listdir(data_dir)]
	fs = random.sample(fs, ntrain+ntest)
	X = pool.map(load_path, fs)
	shapes = [x.shape for x in X]
	trX = X[:-ntest]
	teX = X[-ntest:]
	return trX, teX