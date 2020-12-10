import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # do not delete
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import BayesianRidge

from feature_engineering import feature_eng




# import data
application_test = pd.read_csv('./home-credit-default-risk/application_test.csv')
application_train = pd.read_csv('./home-credit-default-risk/application_train.csv')
bureau = pd.read_csv('./home-credit-default-risk/bureau.csv')
bureau_balance = pd.read_csv('./home-credit-default-risk/bureau_balance.csv')
credit_card_balance = pd.read_csv('./home-credit-default-risk/credit_card_balance.csv')
installments_payments = pd.read_csv('./home-credit-default-risk/installments_payments.csv')
pos_cash = pd.read_csv('./home-credit-default-risk/POS_CASH_balance.csv')
previous_application = pd.read_csv('./home-credit-default-risk/previous_application.csv')

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# dummy the non-numerical columns
non_num_col = application_train.select_dtypes(include=['O']).columns
col_for_dum = non_num_col.drop(['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'EMERGENCYSTATE_MODE'])
train_dummies = pd.get_dummies(application_train, columns=col_for_dum, drop_first=True)
test_dummies = pd.get_dummies(application_test, columns=col_for_dum, drop_first=True)

dum_flag_1 = train_dummies['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})
dum_flag_2 = train_dummies['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})
dum_flag_3 = train_dummies['EMERGENCYSTATE_MODE'].map({'Yes': 1, 'No': 0})
train_dummies['FLAG_OWN_CAR'] = dum_flag_1
train_dummies['FLAG_OWN_REALTY'] = dum_flag_2
train_dummies['EMERGENCYSTATE_MODE'] = dum_flag_3
test_dummies['FLAG_OWN_CAR'] = dum_flag_1
test_dummies['FLAG_OWN_REALTY'] = dum_flag_2
test_dummies['EMERGENCYSTATE_MODE'] = dum_flag_3

# print(train_dummies.shape, test_dummies.shape)
# train_dummies.columns.difference(test_dummies.columns)

# align the train and test datasets
train_tgt = train_dummies['TARGET']
train_dummies, test_dummies = train_dummies.align(test_dummies, join='inner', axis=1)
train_dummies['TARGET'] = train_tgt
# print(train_dummies.shape, test_dummies.shape)

# solve the missing values
irr_col = train_dummies[['SK_ID_CURR', 'TARGET']]
non_tgt = train_dummies.drop(columns=['TARGET'], axis=1)
non_tgt_imputation = non_tgt.loc[:, (non_tgt.nunique() > 1000)]
# print(non_tgt_imputation.columns)
imputer = IterativeImputer(BayesianRidge())
imputed_total = pd.DataFrame(imputer.fit_transform(non_tgt_imputation))
imputed_total.columns = non_tgt_imputation.columns

# outliers
clf = IsolationForest(max_samples=100, random_state=np.random.RandomState(0), contamination=.1)
clf.fit(imputed_total)
if_scores = clf.decision_function(imputed_total)

imputed_total['anomaly'] = clf.predict(imputed_total)
outliers = imputed_total.loc[imputed_total['anomaly'] == -1]
outlier_index = list(outliers.index)
# print(outlier_index)
# print(imputed_total['anomaly'].value_counts())

outlier_ID = list(outliers['SK_ID_CURR'])
X = non_tgt[~non_tgt.SK_ID_CURR.isin(outlier_ID)]
y = irr_col[~irr_col.SK_ID_CURR.isin(outlier_ID)]
# print(X.shape, non_tgt.shape)

# anomaly
# print('DAYS_BIRTH:', '\n', (X['DAYS_BIRTH'] / -365).describe(), '\n')
# print('DAYS_EMPLOYED:', '\n', (X['DAYS_EMPLOYED'] / -365).describe(), '\n')
# print('DAYS_REGISTRATION:', '\n', (X['DAYS_REGISTRATION'] / -365).describe(), '\n')
# print('DAYS_ID_PUBLISH:', '\n', (X['DAYS_ID_PUBLISH'] / -365).describe(), '\n')
# print('DAYS_LAST_PHONE_CHANGE:', '\n', (X['DAYS_LAST_PHONE_CHANGE'] / -365).describe(), '\n')
# print('all states in years')

# handle anomalies in employed state
employed_max = X['DAYS_EMPLOYED'].max()
X['DAYS_EMPLOYED'].replace({employed_max: np.nan}, inplace=True)
test_dummies['DAYS_EMPLOYED'].replace({employed_max: np.nan}, inplace=True)

# checking missing data
total = X.isnull().sum().sort_values(ascending=False)
percent = (X.isnull().sum() / X.isnull().count() * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Missing Total', 'Missing Percent'])
# print(missing_data)

# checking duplicate
non_id_col = [col for col in X.columns if col != 'SK_ID_CURR']
# print('Duplicates:', X[X.duplicated(subset=non_id_col, keep=False)].shape[0])

# checking imbalance
# print(y['TARGET'].value_counts())

# combine data from application_tran/test
train_dummies['Credit_flag'] = train_dummies['AMT_INCOME_TOTAL'] > train_dummies['AMT_CREDIT']
train_dummies['Percent_Days_employed'] = train_dummies['DAYS_EMPLOYED']/train_dummies['DAYS_BIRTH'] * 100
train_dummies['Annuity_as_percent_income'] = train_dummies['AMT_ANNUITY'] / train_dummies['AMT_INCOME_TOTAL'] * 100
train_dummies['Credit_as_percent_income'] = train_dummies['AMT_CREDIT'] / train_dummies['AMT_INCOME_TOTAL'] * 100

test_dummies['Credit_flag'] = test_dummies['AMT_INCOME_TOTAL'] > test_dummies['AMT_CREDIT']
test_dummies['Percent_Days_employed'] = test_dummies['DAYS_EMPLOYED'] / test_dummies['DAYS_BIRTH'] * 100
test_dummies['Annuity_as_percent_income'] = test_dummies['AMT_ANNUITY'] / test_dummies['AMT_INCOME_TOTAL'] * 100
test_dummies['Credit_as_percent_income'] = test_dummies['AMT_CREDIT'] / test_dummies['AMT_INCOME_TOTAL'] * 100

# combine data from bureau
feature_eng(test_dummies, train_dummies)


