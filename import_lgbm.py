import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sasctl import Session
from import_sklearn import import_sklearn

# read LightGBM native model from file
bst = lgb.Booster(model_file='test_lgm.txt')
# bst.feature_name()
# score_X = np.array([[1, 2, 3]])
# bst.predict(score_X)

# make pesudo train/test data
n_col = bst.num_feature()
n_samp = 100
X = pd.DataFrame(np.random.normal(size=(n_samp, n_col)),
                 columns=bst.feature_name())
y = pd.Series(np.random.binomial(1, 0.5, size=n_samp), name='y')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)

# convert LightGBM native model to sklearn model
lgbm = lgb.LGBMClassifier()
lgbm._Booster = bst
lgbm.fitted_ = True
lgbm._n_classes = 2
lgbm._n_features = bst.num_feature()
lgbm._le = lgb.sklearn._LGBMLabelEncoder().fit(y_train)
lgbm._class_map = dict(
    zip(lgbm._le.classes_, lgbm._le.transform(lgbm._le.classes_)))

# compare native model predictions with sklearn model predictions
bst_pred = bst.predict(X_test, raw_score=True)
lgbm_pred = lgbm.predict(X_test, raw_score=True)
np.testing.assert_equal(bst_pred, lgbm_pred)

# Viya connection
USER = 'sasdemo'
PWD = 'sas123'
HOST = '172.26.38.244'
sess = Session(HOST, USER, PWD, protocol='http')
conn = sess.as_swat()

# proj / model metadata
PROJ = 'Python LightGBM Test 1022v1'
MODNAME = 'LightGBM'
MODDESC = ''
MODALGO = 'LightGBM'
MODELER = 'Ryan Ma'
EVENT = 1
MODFOLDER = 'model/LightGBM'

# import into MM
import_sklearn(PROJ, lgbm, MODNAME, MODDESC, MODALGO, MODELER,
              X_train, y_train, X_test, y_test, EVENT, MODFOLDER, conn)

# import pickle
# with open('model/LightGBM/LightGBM.pickle', 'rb') as pFile:
#     clf = pickle.load(pFile)

# clf.predict_proba([[1,2,3]])