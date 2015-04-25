import random
import numpy as np
import pandas as pd
from time import time, sleep
from collections import Counter
from multiprocessing import Pool
from matplotlib import pyplot as plt

from foxhound import ops
from foxhound.inits import Normal
from foxhound.iterators import Linear
from foxhound.models import VariationalNetwork, Network

from load import load, mnist

trX, teX, trY, teY = mnist()

init_fn = Normal(scale=0.025)

model = [
	ops.Input(['x', 28*28]),
	ops.Project(1024, init_fn=init_fn),
	ops.Activation('rectify'),

	ops.Variational(10, init_fn=init_fn),

	ops.Project(1024, init_fn=init_fn),
	ops.Activation('rectify'),

	ops.Project(28*28, init_fn=init_fn),
	ops.Activation('sigmoid')
]

model = VariationalNetwork(model)
model.fit(trX, trX, n_iter=2)

Xs = model.sample(trX[:100])

pX = model.predict_proba(trX[:100])

for x, p in zip(trX, pX):
	plt.subplot(1, 2, 1)
	plt.imshow(x.reshape(28, 28), cmap='gray')
	plt.subplot(1, 2, 2)
	plt.imshow(p.reshape(28, 28), cmap='gray')
	plt.show()

