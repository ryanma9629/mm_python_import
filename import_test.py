# -*-coding:utf-8 -*-
'''
@File    :   import_test.py
@Time    :   2022/02/28 16:46
@Author  :   Ryan Ma
@Version :   1.0
@Contact :   ryanma9629@gmail.com
@Desc    :   None
'''

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from sasctl import Session
from import_sklearn import import_sklearn


USER = 'sasdemo'
PWD = 'sas123'
HOST = '172.26.38.244'
sess = Session(HOST, USER, PWD, protocol='http')
conn = sess.as_swat()

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X.columns = [col.replace(' ', '_') for col in X.columns]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)
clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)

# pred = clf.predict(X_test)
# print(classification_report(y_test, pred))

# Model Registeration
PROJ = 'Breast_Cancer'
MODNAME = 'LR'
MODDESC = ''
MODALGO = 'Logistic Regression'
MODELER = 'Ryan Ma'
EVENT = 1
MODFOLER = 'model/LR'

import_sklearn(PROJ, clf, MODNAME, MODDESC, MODALGO, MODELER,
                X_train, y_train, X_test, y_test, EVENT, MODFOLER, conn)
