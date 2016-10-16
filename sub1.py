import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.metrics import  auc

print('prepare train data...')
train = pd.read_csv('data/mid_off_train.csv')
test = pd.read_csv('data/mid_off_test.csv')

print('preprocess...')
train = train.drop(['User_id', 'Merchant_id'], axis =1)
test = test.drop(['User_id', 'Merchant_id'], axis =1)
train = train[train['label'] != 2]
y = train['label']
train = train.replace({'null' : 0})
test = test.replace({'null' : 0})

print('train...')
clt = GradientBoostingClassifier()
#clt.fit(train.ix[:, 0:-1], y)
score = cross_validation.cross_val_score(clt, train.ix[:, 0:-1], y, cv = 5, scoring='roc_auc')

# print('predict...')
# y_pred = clt.predict_proba(test)
#
# print('generate result...')
# test = pd.read_csv('data/off_test.csv')
# test = test[['User_id', 'Coupon_id', 'Date_received']]
# test['Probability'] = y_pred[:, 1]
# test.to_csv('sub1_gbdt.csv', index = False)