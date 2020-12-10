import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# read data from csv files
for dirname, _, filenames in os.walk('./home-credit-default-risk/'):
    for name in filenames:
        os.path.join(dirname, name)

# control output size
pd.set_option('display.max_rows', 500)

# train, test dataset
test = pd.read_csv('./home-credit-default-risk/application_test.csv')
train = pd.read_csv('./home-credit-default-risk/application_train.csv')
df = pd.concat((train.loc[:, 'NAME_CONTRACT_TYPE': 'AMT_REQ_CREDIT_BUREAU_YEAR'],
                test.loc[:, 'NAME_CONTRACT_TYPE': 'AMT_REQ_CREDIT_BUREAU_YEAR']))


# prepare data before tuning
def before_tuning_detail(df):
    detail = pd.DataFrame()
    detail['Missing Percentage'] = round(df.isnull().sum() / df.shape[0] * 100)
    detail['N unique'] = df.nunique()
    detail['Type'] = df.dtypes
    return detail


# output the missing data before tunning
before_detail = before_tuning_detail(df)
# print(before_detail)

# handle with NaN
for column in df:
    if df[column].dtype == 'object':
        df[column].fillna('N')
    df[column].fillna(10000, inplace=True)

# handle with n unique
for column in df:
    if df[column].nunique() <= 30:
        df[column] = df[column].astype(str)

# encoding with dummies
df = pd.get_dummies(df)


# after tuning detail
def after_tuning_detail(df):
    detail = pd.DataFrame()
    detail['Missing Percentage'] = round(df.isnull().sum() / df.shape[0] * 100, 2)
    detail['N unique'] = df.nunique()
    detail['Type'] = df.dtypes
    return detail


# output after tuning details
after_detail = after_tuning_detail(df)
# print(after_detail)

# feature selection preparation
X_train = df[:train.shape[0]]
X_test = df[train.shape[0]:]
tgt = train.TARGET
X_train['Y'] = tgt
df = X_train
X = df.drop('Y', axis=1)
tgt = df.Y

# xgboost
para = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',                         # better with 'gpu_hist'
        'eta': 0.3,
        'max_depth': 6,
        'learning_rate': 0.01,
        'eval_metric': 'auc',
        'min_child_weight': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'seed': 29,
        'reg_lambda': 0.8,
        'reg_alpha': 0.000001,
        'gamma': 0.1,
        'scale_pos_weight': 1,
        'n_estimators': 500,
        'nthread': -1
}

x_train, x_valid, y_train, y_valid = train_test_split(X, tgt, test_size=0.2, random_state=10)
xgb_train = xgb.DMatrix(x_train, label=y_train)
xgb_valid = xgb.DMatrix(x_valid, label=y_valid)
xgb_test = xgb.DMatrix(X_test)

validation = [(xgb_train, 'train'), (xgb_valid, 'valid')]
times = 10000
model = xgb.train(para, xgb_train, times, validation, early_stopping_rounds=50, maximize=True, verbose_eval=10)
p_test = model.predict(xgb_test)

best = pd.DataFrame({'SK_ID_CURR': test['SK_ID_CURR', 'TARGET': p_test]})
print(best.shape)


