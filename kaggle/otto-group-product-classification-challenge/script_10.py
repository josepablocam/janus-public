# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/train.csv")
test_df=pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/test.csv")
train_df
train_df["target"].unique()
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
train_df["target"]=le.fit_transform(train_df["target"])
train_df["target"].unique()
import seaborn as sns
sns.distplot(train_df["target"])
train_df.isnull().sum()
train_df.skew()
target=train_df["target"]
del train_df["target"]
del train_df["id"]
target
for i in train_df.columns.values:
    train_df[i]=np.sqrt(train_df[i])
train_df
train_df.skew()
test_df
del test_df["id"]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train_df,target,random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
k_fold=KFold(n_splits=10,shuffle= True,random_state= 0)


clf = RandomForestClassifier(n_estimators = 100)

from sklearn.metrics import log_loss
clf.fit(X_train,y_train)
clf_probs=clf.predict_proba(X_test)
score=log_loss(y_test,clf_probs)
score
print(clf.score(X_test,y_test)*100)
print(clf.score(X_train,y_train)*100)
prediction=clf.predict_proba(test_df)
sample=pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/sampleSubmission.csv")
prediction = pd.DataFrame(prediction, index=sample.id.values, columns=sample.columns[1:])
prediction
prediction.to_csv('sub.csv', index_label='id')
submission=pd.read_csv("sub.csv")
submission.head(10)

