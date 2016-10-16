import pandas as pd

train = pd.read_csv('data/off_train.csv')
user_list = train['User_id'].drop_duplicates()
user_list = pd.DataFrame({'User_id' : user_list, 'index' : range(0, user_list.__len__())})
user_list.to_csv('data/index_user.csv', index = False)
merchant_list = train['Merchant_id'].drop_duplicates()
merchant_list = pd.DataFrame({'Merchant_id' : merchant_list, 'index' : range(0, merchant_list.__len__())})
merchant_list.to_csv('data/index_merchant.csv', index = False)

train = pd.read_csv('data/off_test.csv', chunksize = 900000)