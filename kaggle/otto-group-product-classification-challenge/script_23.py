import sys
sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]

import cuml.manifold    as tsne_rapids
import sklearn.manifold as tsne_sklearn
train = pd.read_csv('../input/otto-group-product-classification-challenge/train.csv')
train = train.sample(15000)
train.shape
train.head()
y = np.array( [int(v.split('_')[1]) for v in train.target.values ] )
train.drop( ['id','target'], inplace=True, axis=1 )
tsne = tsne_sklearn.TSNE(n_components=2, random_state=2020 )
train_2D_sklearn = tsne.fit_transform( train.values )
plt.scatter(train_2D_sklearn[:,0], train_2D_sklearn[:,1], c = y, s = 0.5)
tsne = tsne_rapids.TSNE(n_components=2, perplexity=30, random_state=2020 )
train_2D_rapids = tsne.fit_transform( train.values )
plt.scatter(train_2D_rapids[:,0], train_2D_rapids[:,1], c = y, s = 0.5)
