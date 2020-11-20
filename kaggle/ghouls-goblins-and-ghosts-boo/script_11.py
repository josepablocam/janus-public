import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

monster_train = pd.read_csv('../input/train.csv', index_col='id')
monster_test = pd.read_csv('../input/test.csv', index_col='id')
monster_train.head()
sns.pairplot(monster_train, size=1.5, hue='type')
msk = np.random.rand(len(monster_train)) < 0.8
monster_train_A = monster_train[msk]
monster_train_B = monster_train[~msk]

print('%d monsters in A' % len(monster_train_A))
print('%d monsters in B' % len(monster_train_B))
features = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']

clf = KNeighborsClassifier(n_neighbors=6)
clf.fit(monster_train_A[features], monster_train_A['type'])

preds = clf.predict(monster_train_B[features])
accuracy = np.where(preds==monster_train_B['type'], 1, 0).sum() / float(len(monster_train_B))
print(accuracy)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(monster_train_A[features], monster_train_A['type'])

preds_d = clf.predict(monster_train_B[features])
accuracy = np.where(preds_d==monster_train_B['type'], 1, 0).sum() / float(len(monster_train_B))
print(accuracy)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, n_estimators=100, max_features=1)
clf.fit(monster_train_A[features], monster_train_A['type'])

preds_d = clf.predict(monster_train_B[features])
accuracy = np.where(preds_d==monster_train_B['type'], 1, 0).sum() / float(len(monster_train_B))
print(accuracy)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
clf = GaussianProcessClassifier(4.0 * RBF(1.0), warm_start=True, n_jobs=-1)
clf.fit(monster_train_A[features], monster_train_A['type'])

preds_d = clf.predict(monster_train_B[features])
accuracy = np.where(preds_d==monster_train_B['type'], 1, 0).sum() / float(len(monster_train_B))
print(accuracy)
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(alpha=1, max_iter=400)
clf.fit(monster_train_A[features], monster_train_A['type'])

preds_d = clf.predict(monster_train_B[features])
accuracy = np.where(preds_d==monster_train_B['type'], 1, 0).sum() / float(len(monster_train_B))
print(accuracy)
