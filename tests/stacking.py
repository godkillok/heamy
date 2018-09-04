from heamy.dataset import Dataset
from heamy.estimator import  Classifier
from heamy.pipeline import ModelsPipeline
from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from xgboost.sklearn import XGBClassifier
from sklearn import svm

import scipy
from sklearn.metrics import accuracy_score,f1_score
#加载数据集
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import tarfile
import codecs
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
data = fetch_20newsgroups()
X, y = data.data, data.target
print(len(X))


vec = TfidfVectorizer(ngram_range=(1,1),min_df=3, max_df=0.8,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(X)
X_train, X_test, y_train, y_test =train_test_split(trn_term_doc, y, test_size=0.1, random_state=111)
# X_train=X_train.toarray()
# X_test=X_test.toarray()
print(type(X_train))
#创建数据集1
dataset = Dataset(X_train,y_train,X_test,use_cache=False)
class_use_cache=False
#创建RF模型和LR模型
model_nb = Classifier(dataset=dataset, estimator=MultinomialNB,name='nb',use_cache=class_use_cache)
model_lr = Classifier(dataset=dataset, estimator=LogisticRegression, parameters={'C':4, 'dual':True,'n_jobs':-1},name='lr',use_cache=class_use_cache)
model_svm = Classifier(dataset=dataset, estimator=svm.SVC, parameters={ 'probability':True},name='svm',use_cache=class_use_cache)
model_per = Classifier(dataset=dataset, estimator=Perceptron, parameters={ 'n_iter':50,'penalty':'l2','n_jobs':-1},name='Perceptron',use_cache=class_use_cache)
model_svc = Classifier(dataset=dataset, estimator=svm.LinearSVC,name="LinearSVC",use_cache=class_use_cache)
model_svc = Classifier(dataset=dataset, estimator=svm.LinearSVC,name="LinearSVC",use_cache=class_use_cache)


# Stack两个模型mhg
# Returns new dataset with out-of-fold predictionmodel_svc,
pipeline = ModelsPipeline(model_nb)
stack_ds = pipeline.stack(k=3,seed=111)
print(stack_ds.X_train.shape)
#第二层使用lr模型stack
stacker = Classifier(dataset=stack_ds, estimator=LogisticRegression, parameters={'C':4, 'dual':True,'n_jobs':-1},use_cache=False,probability=False)
results = stacker.predict()
result_list=list(results)



print(accuracy_score(y_test, results))
# 使用10折交叉验证结果
results10 = stacker.validate(k=3,scorer=accuracy_score)
print(results10)