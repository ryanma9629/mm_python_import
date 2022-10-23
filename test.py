import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

dat = pd.read_csv('d:/data/hmeq_imp.csv')
y = dat['BAD']
X = dat.drop('BAD', axis=1)
col_cat = X.columns[X.dtypes=='object']

le = LabelEncoder()
for c in col_cat:
    X.loc[:, c] = le.fit_transform(X[c])

lgtrain = lgb.Dataset(X, label=y)
lgbm = lgb.train(params={}, train_set=lgtrain)

lgbm.save_model()

dat = lgb.Dataset()