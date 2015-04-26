import random
import numpy as np
import pandas as pd
from time import time, sleep
from collections import Counter
from multiprocessing import Pool
from matplotlib import pyplot as plt
from scipy.misc import imsave

from foxhound import ops
from foxhound.inits import Normal
from foxhound.updates import NAG, Adam
from foxhound.iterators import Linear
from foxhound.models import VariationalNetwork, Network
from foxhound.vis import color_grid_vis

from load import github_avatars, mnist

# trX, teX, trY, teY = mnist()

trX, teX = github_avatars(ntrain=10000, ntest=1000)

trX = np.asarray(trX, dtype=np.uint8).reshape(-1, 64*64*3)
teX = np.asarray(teX, dtype=np.uint8).reshape(-1, 64*64*3)

init_fn = Normal(scale=0.01)
# update_fn = NAG(lr=0.0001)
update_fn = Adam()

n_code = 10
n_hidden = 1024

model = [
	ops.Input(['x', 64*64*3]),
	ops.Project(n_hidden, init_fn=init_fn, update_fn=update_fn),
	ops.Activation('rectify'),
	ops.Project(n_hidden, init_fn=init_fn, update_fn=update_fn),
	ops.Activation('tanh'),

	ops.Variational(n_code, init_fn=init_fn, update_fn=update_fn),

	ops.Project(n_hidden, init_fn=init_fn, update_fn=update_fn),
	ops.Activation('rectify'),
	ops.Project(n_hidden, init_fn=init_fn, update_fn=update_fn),
	ops.Activation('tanh'),

	ops.Project(64*64*3, init_fn=init_fn, update_fn=update_fn),
	ops.Activation('sigmoid')
]

Zs = np.random.normal(size=(100, n_code))
model = VariationalNetwork(model, verbose=1)
for i in range(100):
	model.fit(trX/127.5 - 1., trX/255., n_iter=1)

	sX = model.sample(Zs)

	# pX = model.predict_proba(trX[:100])

	# for x, p in zip(trX, pX):
	# 	plt.subplot(1, 2, 1)
	# 	plt.imshow(x.reshape(28, 28), cmap='gray')
	# 	plt.subplot(1, 2, 2)
	# 	plt.imshow(p.reshape(28, 28), cmap='gray')
	# 	plt.show()

	img = color_grid_vis(sX, transform=lambda x:x.reshape(64, 64, 3), show=False)
	imsave('vis/%d.png'%i, img)
