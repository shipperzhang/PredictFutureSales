import numpy as np
import pandas as pd

import os

train = pd.read_csv('./data/sales_train.csv.gz')
test = pd.read_csv('./data/test.csv.gz')
submission = pd.read_csv('./data/sample_submission.csv.gz')
items = pd.read_csv('./data/items.csv')
item_cats = pd.read_csv('./data/item_categories.csv')
shops = pd.read_csv('./data/shops.csv')
print(train.shape)
train = train.loc[train['item_cnt_day'] >= -1.0].loc[train['item_cnt_day'] <= 20.0].loc[train['item_price'] <= 1000.0].loc[train['item_price'] >= 0]
print(train.shape)
# print(train.describe())
train = train.groupby(["date_block_num","shop_id", "item_id"])
train = train.aggregate({"item_price":np.mean, "item_cnt_day":np.sum}).fillna(0)
train.reset_index(level=["date_block_num", "shop_id", "item_id"], inplace=True)
train['item_cnt_day'] = train['item_cnt_day'].clip(0,20)
print(train.describe())

# for index, row in train.iterrows():
#     print(row['date_block_num'],row['shop_id'],row['item_id'])
#     break

# df_m = train.groupby(["date_block_num","shop_id", "item_id"])
# month_sum = df_m.aggregate({"item_price":np.mean, "item_cnt_day":np.sum}).fillna(0)
# month_sum.reset_index(level=["date_block_num", "shop_id", "item_id"], inplace=True)
# # month_sum = month_sum.rename(columns={ month_sum.columns[4]: "item_cnt_month"})
# print(month_sum.shape)
# print(month_sum.describe())
# tmp = train.loc[train['item_cnt_day'] > 166.0]
# tmp = train.loc[train['item_id']==20949.0].loc[train['shop_id']==31.0]
# print(tmp)
# print(tmp.shape)



# new_submission = pd.merge(month_sum, test, how='right', left_on=['shop_id','item_id'], right_on = ['shop_id','item_id']).fillna(0)
# new_submission.drop(['shop_id', 'item_id'], axis=1)
# new_submission = new_submission[['ID','item_cnt_month']]
# new_submission['item_cnt_month'] = new_submission['item_cnt_month'].clip(0,20)

