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
user = 'sasdemo'
pwd = 'sas123'
host = '172.26.38.244'
viya_sess = Session(host, user, pwd, protocol='http')
viya_conn = viya_sess.as_swat()

# proj / model metadata
proj = 'Python LightGBM Test 1022v2'
mod_name = 'LightGBM1022v2'
mod_desc = 'A LightGBM model imported from file.'
mod_algo = 'LightGBM'
mod_owner = 'Ryan Ma'
target_event = 1
mod_folder = 'model/LightGBM'

# import into MM
import_sklearn(viya_conn,
               mod_folder,
               proj,
               lgbm,
               mod_name,
               mod_desc,
               mod_algo,
               mod_owner,
               target_event,
               X_train,
               y_train,
               X_test,
               y_test)


# import pickle
# with open('model/LightGBM/LightGBM.pickle', 'rb') as pFile:
#     clf = pickle.load(pFile)

# clf.predict_proba([[1,2,3]])
