import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # do not delete
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import BayesianRidge

from feature_engineering import feature_eng
from info_value import iv_calculator

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score




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

# combine data from other csv files
train, test = feature_eng(test_dummies, train_dummies)

# train, test and validation sets
X = train.drop(columns=['TARGET'])
y = train['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# calculate information value
final_iv, IV = iv_calculator(X_train, y_train)
# print(IV)

list_of_columns=IV[IV['IV'] > 0.02]['VAR_NAME'].to_list()
# print(len(list_of_columns))

X_train_selected_features = X_train[list_of_columns]
X_test_selected_features = X_test[list_of_columns]
X_train_selected_features['SK_ID_CURR'] = X_train['SK_ID_CURR']
X_test_selected_features['SK_ID_CURR'] = X_test['SK_ID_CURR']

test_selected_features = test[list_of_columns]
test_selected_features['SK_ID_CURR'] = test['SK_ID_CURR']

# data imputation
imputer = IterativeImputer(BayesianRidge())
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_selected_features))
X_train_imputed.columns = X_train_selected_features.columns

imputer = IterativeImputer(BayesianRidge())
test_selected_features_subset1 = test_selected_features.iloc[:, np.r_[63, 0: 30]]
test_imputed_subset1 = pd.DataFrame(imputer.fit_transform(test_selected_features_subset1))
test_imputed_subset1.columns = test_selected_features_subset1.columns

test_selected_features_subset2 = test_selected_features.iloc[:, np.r_[63, 31: 63]]
test_imputed_subset2 = pd.DataFrame(imputer.fit_transform(test_selected_features_subset2))
test_imputed_subset2.columns = test_selected_features_subset2.columns

test_imputed = pd.merge(test_imputed_subset1, test_imputed_subset2, on='SK_ID_CURR')

imputer = IterativeImputer(BayesianRidge())
X_test_imputed = pd.DataFrame(imputer.fit_transform(X_test_selected_features))
X_test_imputed.columns = X_test_selected_features.columns

# print(X_test_imputed.shape)
# print(X_train_imputed.shape)
# print(test_imputed.shape)

# X_train_imputed, test_imputed = test_imputed.align(X_train_imputed, join='inner', axis=1)
# print(X_train_imputed.shape)
# print(y_train.shape)
# X_train_imputed, X_test_imputed = X_train_imputed.align(X_test_imputed, join='inner', axis=1)
# print(X_train_imputed.shape)
# print(X_test_imputed.shape)


# ML part
# logistic regression
lr_clf = LogisticRegression(random_state=0, class_weight='balanced')
lr_clf.fit(X_train_imputed, y_train)
y_train_pred_lr = cross_val_predict(lr_clf, X_train_imputed, y_train, cv=3)
print('lr_accuracy(Training):', cross_val_score(lr_clf, X_train_imputed, y_train, cv=3, scoring='accuracy'))
print('lr_accuracy(Test):', cross_val_score(lr_clf, X_test_imputed, y_test, cv=3, scoring='accuracy'))

# random forest
rf_clf = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1, class_weight="balanced")
rf_clf.fit(X_train_imputed, y_train)
print('rf_accuracy(Training):', cross_val_score(rf_clf, X_train_imputed, y_train, cv=3, scoring='accuracy'))
print('rf_accuracy(Test):', cross_val_score(rf_clf, X_test_imputed, y_test, cv=3, scoring='accuracy'))

# xgboost
weight = y_train.value_counts().values.tolist()[0] / y_train.value_counts().values.tolist()[1]
xgb_clf = XGBClassifier(scale_pos_weight=weight)
xgb_clf.fit(X_train_imputed, y_train)
print('xgb_accuracy(Training):', cross_val_score(xgb_clf, X_train_imputed, y_train, cv=3, scoring='accuracy'))
print('xgb_accuracy(Test):', cross_val_score(xgb_clf, X_test_imputed, y_test, cv=3, scoring='accuracy'))


# # result
# lr_accuracy(Training): [0.64089119 0.64428131 0.62870418]
# lr_accuracy(Test): [0.65265109 0.64362714 0.64011512]
# rf_accuracy(Training): [0.91873468 0.9189176  0.91872149]
# rf_accuracy(Test): [0.91829667 0.91854056 0.918004  ]
# xgb_accuracy(Training): [0.74964331 0.7494238  0.74883539]
# xgb_accuracy(Test): [0.82425248 0.82083801 0.82654505]

