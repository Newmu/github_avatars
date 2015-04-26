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

def github_avatars(ntrain=2000, ntest=1000):
	pool = Pool(8)
	fs = [os.path.join(data_dir, path) for path in os.listdir(data_dir)]
	fs = random.sample(fs, ntrain+ntest)
	X = pool.map(load_path, fs)
	shapes = [x.shape for x in X]
	trX = X[:-ntest]
	teX = X[-ntest:]
	return trX, teX

def mnist(ntrain=60000, ntest=10000, onehot=False):
	datasets_dir = '/home/alec/datasets/'
	data_dir = os.path.join(datasets_dir,'mnist')
	fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000, -1)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000, 1))

	fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000, -1)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000, 1))

	trX = trX/255.
	teX = teX/255.

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		trY = one_hot(trY)
		teY = one_hot(teY)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

	return trX, teX, trY, teY
