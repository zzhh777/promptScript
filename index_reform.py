#note:this is the script cut that shouldn't run directly
import pandas as pd
import numpy as np

def rate_reform(x):
    if x[3] == 'null':
        #print(x)
        return x
    try:
        x[3] = float(x[3])
        x[6] = 0
    except:
        s = x[3].split(':')
        #print(s)
        x[3] = float(s[1]) / float(s[0])
        x[6] = 1
    return x

def label(x):
    if (x[6] == 'null') & (x[2] != 'null'):
        return 0
    elif (x[6] != 'null') & (x[2] != 'null'):
        return 1
    else:
        return 2

def replace_id(x, dic_u, dic_m):
    x[0] = dic_u[x[0]]
    x[1] = dic_m[x[1]]
    return x

def replace_id_1(x, dic_m):
    x[0] = dic_m[x[0]]
    return x

print('prepare train data...')
train = pd.read_csv('data/my_train.csv')
train = train[['User_id', 'Merchant_id', 'Distance', 'Discount_rate', 'label']]
train = train[train['label'] != 2]
y = train[['label']]

#replace id
dic_u = dict(zip(train['User_id'].drop_duplicates(), range(0, train.__len__())))
dic_m = dict(zip(train['Merchant_id'].drop_duplicates(), range(0, train.__len__())))
train = train.apply(replace_id, dic_u = dic_u, dic_m = dic_m, axis = 1)

