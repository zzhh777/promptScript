import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.metrics import  auc

print('prepare train data...')
train = pd.read_csv('data/mid_off_train.csv')
test = pd.read_csv('data/mid_off_test.csv')

print('preprocess...')
train = train.replace({'null' : 0})
test = test.replace({'null' : 0})
#user action number
act_num = train[train['label'] != 0].groupby('User_id').count()[['Merchant_id']]
act_num['act_num'] = act_num['Merchant_id']
act_num['User_id'] = act_num.index
act_num = act_num.drop(['Merchant_id'], axis = 1)
train = pd.merge(train, act_num, how = 'left', on = 'User_id')
test = pd.merge(test, act_num, how = 'left', on = 'User_id')

#coupon number the merchant pushed
mer_hot = train.groupby('Merchant_id').count()[['User_id']]
mer_hot['Merchant_id'] = mer_hot.index
mer_hot['mer_hot'] = mer_hot['User_id']
mer_hot = mer_hot.drop(['User_id'], axis = 1)
train = pd.merge(train, mer_hot, how = 'left', on = 'Merchant_id')
test = pd.merge(test, mer_hot, how = 'left', on = 'Merchant_id')

mer_use = train[train['label'] != 0].groupby('Merchant_id').count()[['User_id']]
mer_use['Merchant_id'] = mer_use.index
mer_use['mer_use'] = mer_use['User_id']
mer_use = mer_use.drop(['User_id'], axis = 1)
train = pd.merge(train, mer_use, how = 'left', on = 'Merchant_id')
test = pd.merge(test, mer_use, how = 'left', on = 'Merchant_id')
#fill missing data and generate train, test data
test = test.fillna(0)
train = train.fillna(0)
train = train[train['label'] != 2]
y = train['label']
train = train[['Distance', 'Discount_rate', 'dis_type', 'act_num', 'mer_hot', 'mer_use']]
test = test[['Distance', 'Discount_rate', 'type', 'act_num', 'mer_hot', 'mer_use']]


print('train...')
clt = GradientBoostingClassifier()
clt.fit(train, y)
score = cross_validation.cross_val_score(clt, train.ix[:, 0:-1], y, cv = 5, scoring = 'roc_auc')
print(score)

print('predict...')
y_pred = clt.predict_proba(test)
#
# print('generate result...')
# test = pd.read_csv('data/off_test.csv')
# test = test[['User_id', 'Coupon_id', 'Date_received']]
# test['Probability'] = y_pred[:, 1]
# test.to_csv('sub2_gbdt.csv', index = False)