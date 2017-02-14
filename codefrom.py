#MODULES
import numpy as np
import math
import random
import csv
import pandas as pd
import statsmodels.api as st
import matplotlib.pyplot as plt
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.tree import DecisionTreeRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score


stemmer = SnowballStemmer('english')

#DATA
df_train = pd.read_csv('train1.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('trial.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('product_descriptions.csv', encoding="ISO-8859-1")
#print(df_test)
num_train = df_train.shape[0]

def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

#taking all data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))

df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']

df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)

#setting parameters
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']

#distribution of data
y_train_int=[]
y_test=df_test['relevance'].values
y_test_int=df_test['relevance'].values.astype(int)
x_plot= df_train['id'].values
y_train = df_train['relevance'].values
y_train_int=df_train['relevance'].values.astype(int)
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values
#print (y_test)

#Different types of Regressor:-
# RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
rfr.fit(X_train,y_train)
y_pred_RandomForestRegressor = rfr.predict(X_test)
print("RANDOM FOREST REGRESSOR")
print("Accuracy:",rfr.score(X_test,y_test))
print("MEAN SQ ERROR:",mean_squared_error(y_test,rfr.predict(X_test)))


#BaggingRegressor
br = BaggingRegressor(n_estimators=45, max_samples=0.1, random_state=25)
br.fit(X_train, y_train)
y_pred_BaggingRegressor = br.predict(X_test)
print("BAGGING REGRESSOR")
print("Accuracy:",br.score(X_test,y_test))
print("MEAN SQ ERROR:",mean_squared_error(y_test,br.predict(X_test)))


#DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X_train, y_train)
y_pred_DecisionTreeRegressor = dtr.predict(X_test)
print("DECISION TREE REGRESSOR")
print("Accuracy:",dtr.score(X_test,y_test))
print("MEAN SQ ERROR:",mean_squared_error(y_test,dtr.predict(X_test)))


# K-NEIGHBORS Regressor
nr = KNeighborsRegressor(n_neighbors=3)
nr.fit(X_train, y_train)
y_pred_KNeighborsRegressor=nr.predict(X_test)
print("K-NEIGHBORS Regressor")
print("Accuracy:",nr.score(X_test,y_test))
print("MEAN SQ ERROR:",mean_squared_error(y_test,nr.predict(X_test)))

'''
#CLASSIFIERS:-
#Applying the method DecisionTreeClassifier
dts = DecisionTreeClassifier(random_state=0)
dts.fit(X_train, y_train)
y_DecisionTreeClassifier = dts.predict(X_test)
#print("DecisionTreeClassifier")
print("Accuracy:",dts.score(X_test,y_test))
print("MEAN SQ ERROR:",mean_squared_error(y_test,dts.predict(X_test)))
'''

#BaggingClassifier
bagging = BaggingClassifier(n_estimators=45,max_samples=0.5, max_features=0.5)
bagging.fit(X_train, y_train_int)
y_BaggingClassifier = bagging.predict(X_test)
print("BaggingClassifier")
print("Accuracy:",bagging.score(X_test,y_test_int))
print("MEAN SQ ERROR:",mean_squared_error(y_test_int,bagging.predict(X_test)))

#KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(X_train, y_train_int) 
y_pred_KNeighborsClassifier = neigh.predict(X_test)
print("KNeighborsClassifier")
print("Accuracy:",neigh.score(X_test,y_test_int))
print("MEAN SQ ERROR:",mean_squared_error(y_test_int,neigh.predict(X_test)))

#BAGGING META-ESTIMATORS
#with random forest
rf_b = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf_rf = BaggingRegressor(rf_b, n_estimators=45, max_samples=0.1, random_state=25)
clf_rf.fit(X_train, y_train)
y_b_rf = clf_rf.predict(X_test)
print("BAGGING+RANDOMFOREST")
print("Accuracy:",clf_rf.score(X_test,y_test))
print("MEAN SQ ERROR:",mean_squared_error(y_test,clf_rf.predict(X_test)))


