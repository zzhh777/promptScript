import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.metrics import  auc

print('prepare train data...')
train = pd.read_csv('data/mid_off_train.csv')
test = pd.read_csv('data/mid_off_test.csv')
train_on = pd.read_csv('data/mid_on_train.csv')

print('preprocess...')
train = train.replace({'null' : 0})
test = test.replace({'null' : 0})
#user off line feature:user action number
offuser_buy_num = train[train['label'] != 0].groupby('User_id').count()[['Merchant_id']]
offuser_buy_num['offuser_buy_num'] = offuser_buy_num['Merchant_id']
offuser_buy_num['User_id'] = offuser_buy_num.index
offuser_buy_num = offuser_buy_num.drop(['Merchant_id'], axis = 1)
train = pd.merge(train, offuser_buy_num, how = 'left', on = 'User_id')
test = pd.merge(test, offuser_buy_num, how = 'left', on = 'User_id')

#user off line feature:rate of user use coupon in all action
train_labels = train[['label', 'User_id']]
train_labels = train_labels.replace({2 : 1})
offuser_coupon_rate = train_labels.groupby('User_id').mean()[['label']]
offuser_coupon_rate['User_id'] = offuser_coupon_rate.index
offuser_coupon_rate['offuser_coupon_rate'] = offuser_coupon_rate['label']
offuser_coupon_rate = offuser_coupon_rate.drop(['label'], axis = 1)
train = pd.merge(train, offuser_coupon_rate, how = 'left', on = 'User_id')
test = pd.merge(test, offuser_coupon_rate, how = 'left', on = 'User_id')

#user off line feature:rate of user user coupon when buy sth.
train_labels = train[['label', 'User_id']]
train_labels = train_labels[train_labels['label'] != 0]
train_labels['label'] = 1 - (train_labels['label'] - 1)
offuser_coupon_rate_buy = train_labels.groupby('User_id').mean()[['label']]
offuser_coupon_rate_buy['User_id'] = offuser_coupon_rate_buy.index
offuser_coupon_rate_buy['offuser_coupon_rate_buy'] = offuser_coupon_rate_buy['label']
offuser_coupon_rate_buy = offuser_coupon_rate_buy.drop(['label'], axis = 1)
train = pd.merge(train, offuser_coupon_rate_buy, how = 'left', on = 'User_id')
test = pd.merge(test, offuser_coupon_rate_buy, how = 'left', on = 'User_id')

#merchant off line feature:coupon number the merchant pushed
offmer_coupon_num = train.groupby('Merchant_id').count()[['User_id']]
offmer_coupon_num['Merchant_id'] = offmer_coupon_num.index
offmer_coupon_num['offmer_coupon_num'] = offmer_coupon_num['User_id']
offmer_coupon_num = offmer_coupon_num.drop(['User_id'], axis = 1)
train = pd.merge(train, offmer_coupon_num, how = 'left', on = 'Merchant_id')
test = pd.merge(test, offmer_coupon_num, how = 'left', on = 'Merchant_id')

#merchant off line feature:coupon number that used of merchant
offmer_coupon_use = train[train['label'] != 0].groupby('Merchant_id').count()[['User_id']]
offmer_coupon_use['Merchant_id'] = offmer_coupon_use.index
offmer_coupon_use['offmer_coupon_use'] = offmer_coupon_use['User_id']
offmer_coupon_use = offmer_coupon_use.drop(['User_id'], axis = 1)
train = pd.merge(train, offmer_coupon_use, how = 'left', on = 'Merchant_id')
test = pd.merge(test, offmer_coupon_use, how = 'left', on = 'Merchant_id')

#fill missing data and generate train, test data
test = test.fillna(0)
train = train.fillna(0)
train = train[train['label'] != 2]
y = train['label']
train = train[['Distance', 'Discount_rate', 'offuser_buy_num', 'offmer_coupon_num',\
               'offmer_coupon_use', 'offuser_coupon_rate', 'offuser_coupon_rate_buy']]
test = test[['Distance', 'Discount_rate', 'offuser_buy_num', 'offmer_coupon_num',\
             'offmer_coupon_use', 'offuser_coupon_rate', 'offuser_coupon_rate_buy']]

print('train...')
clt = GradientBoostingClassifier()
clt.fit(train, y)
#score = cross_validation.cross_val_score(clt, train, y, cv = 3, scoring = 'roc_auc')
#print('score:')
#print(score)
#print('feature:')
#print(clt.feature_importances_)

print('predict...')
y_pred = clt.predict_proba(test)

print('generate result...')
test = pd.read_csv('data/off_test.csv')
test = test[['User_id', 'Coupon_id', 'Date_received']]
test['Probability'] = y_pred[:, 1]
test.to_csv('sub3_gbdt.csv', index = False)