import pandas as pd
import numpy as np
import gc


def feature_eng(test_dummies, train_dummies):
    # import
    application_test = pd.read_csv('./home-credit-default-risk/application_test.csv')
    application_train = pd.read_csv('./home-credit-default-risk/application_train.csv')
    bureau = pd.read_csv('./home-credit-default-risk/bureau.csv')
    bureau_balance = pd.read_csv('./home-credit-default-risk/bureau_balance.csv')
    credit_card_balance = pd.read_csv('./home-credit-default-risk/credit_card_balance.csv')
    installments_payments = pd.read_csv('./home-credit-default-risk/installments_payments.csv')
    pos_cash = pd.read_csv('./home-credit-default-risk/POS_CASH_balance.csv')
    previous_application = pd.read_csv('./home-credit-default-risk/previous_application.csv')

    # combine data from application_tran/test with FE
    train_dummies['Credit_flag'] = train_dummies['AMT_INCOME_TOTAL'] > train_dummies['AMT_CREDIT']
    train_dummies['Percent_Days_employed'] = train_dummies['DAYS_EMPLOYED'] / train_dummies['DAYS_BIRTH'] * 100
    train_dummies['Annuity_as_percent_income'] = train_dummies['AMT_ANNUITY'] / train_dummies['AMT_INCOME_TOTAL'] * 100
    train_dummies['Credit_as_percent_income'] = train_dummies['AMT_CREDIT'] / train_dummies['AMT_INCOME_TOTAL'] * 100

    test_dummies['Credit_flag'] = test_dummies['AMT_INCOME_TOTAL'] > test_dummies['AMT_CREDIT']
    test_dummies['Percent_Days_employed'] = test_dummies['DAYS_EMPLOYED'] / test_dummies['DAYS_BIRTH'] * 100
    test_dummies['Annuity_as_percent_income'] = test_dummies['AMT_ANNUITY'] / test_dummies['AMT_INCOME_TOTAL'] * 100
    test_dummies['Credit_as_percent_income'] = test_dummies['AMT_CREDIT'] / test_dummies['AMT_INCOME_TOTAL'] * 100

    # combine data from bureau with FE
    # numerical
    grp = bureau.drop(['SK_ID_BUREAU'], axis=1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    grp.columns = ['BUREAU_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]
    app_bureau = train_dummies.merge(grp, on='SK_ID_CURR', how='left')
    app_bureau.update(app_bureau[grp.columns].fillna(0))

    app_bureau_test = test_dummies.merge(grp, on='SK_ID_CURR', how='left')
    app_bureau_test.update(app_bureau_test[grp.columns].fillna(0))

    # categorical
    bureau_categorical = pd.get_dummies(bureau.select_dtypes('object'))
    bureau_categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
    grp = bureau_categorical.groupby(by=['SK_ID_CURR']).mean().reset_index()
    grp.columns = ['BUREAU_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]
    app_bureau = app_bureau.merge(grp, on='SK_ID_CURR', how='left')
    app_bureau.update(app_bureau[grp.columns].fillna(0))

    app_bureau_test = app_bureau_test.merge(grp, on='SK_ID_CURR', how='left')
    app_bureau_test.update(app_bureau_test[grp.columns].fillna(0))

    # FE
    # loans per customer
    grp = bureau.groupby(by=['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index()\
        .rename(columns={'SK_ID_BUREAU': 'BUREAU_LOAN_COUNT'})
    app_bureau = app_bureau.merge(grp, on='SK_ID_CURR', how='left')
    app_bureau['BUREAU_LOAN_COUNT'] = app_bureau['BUREAU_LOAN_COUNT'].fillna(0)
    app_bureau_test = app_bureau_test.merge(grp, on='SK_ID_CURR', how='left')
    app_bureau_test['BUREAU_LOAN_COUNT'] = app_bureau_test['BUREAU_LOAN_COUNT'].fillna(0)

    # types of loan per customer
    grp = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by=['SK_ID_CURR'])['CREDIT_TYPE']\
        .nunique().reset_index().rename(columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})
    app_bureau = app_bureau.merge(grp, on='SK_ID_CURR', how='left')
    app_bureau['BUREAU_LOAN_TYPES'] = app_bureau['BUREAU_LOAN_TYPES'].fillna(0)
    app_bureau_test = app_bureau_test.merge(grp, on='SK_ID_CURR', how='left')
    app_bureau_test['BUREAU_LOAN_TYPES'] = app_bureau_test['BUREAU_LOAN_TYPES'].fillna(0)

    # debt / credit
    bureau['AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM'].fillna(0)
    bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)
    grp1 = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM']\
        .sum().reset_index().rename(columns={'AMT_CREDIT_SUM': 'TOTAL_CREDIT_SUM'})
    grp2 = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT']\
        .sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CREDIT_SUM_DEBT'})
    grp1['DEBT_CREDIT_RATIO'] = grp2['TOTAL_CREDIT_SUM_DEBT'] / grp1['TOTAL_CREDIT_SUM']

    del grp1['TOTAL_CREDIT_SUM']

    app_bureau = app_bureau.merge(grp1, on='SK_ID_CURR', how='left')
    app_bureau['DEBT_CREDIT_RATIO'] = app_bureau['DEBT_CREDIT_RATIO'].fillna(0)
    app_bureau['DEBT_CREDIT_RATIO'] = app_bureau.replace([np.inf, -np.inf], 0)
    app_bureau['DEBT_CREDIT_RATIO'] = pd.to_numeric(app_bureau['DEBT_CREDIT_RATIO'], downcast='float')

    app_bureau_test = app_bureau_test.merge(grp1, on='SK_ID_CURR', how='left')
    app_bureau_test['DEBT_CREDIT_RATIO'] = app_bureau_test['DEBT_CREDIT_RATIO'].fillna(0)
    app_bureau_test['DEBT_CREDIT_RATIO'] = app_bureau_test.replace([np.inf, -np.inf], 0)
    app_bureau_test['DEBT_CREDIT_RATIO'] = pd.to_numeric(app_bureau_test['DEBT_CREDIT_RATIO'], downcast='float')
    print((app_bureau[app_bureau['DEBT_CREDIT_RATIO'] > 0.5]['TARGET'].value_counts()
     / len(app_bureau[app_bureau['DEBT_CREDIT_RATIO'] > 0.5])) * 100)

    # Overdue / debt
    bureau['AMT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0)
    bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)
    grp1 = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE']\
        .sum().reset_index().rename(columns={'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})
    grp2 = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT']\
        .sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    grp1['OVERDUE_DEBT_RATIO'] = grp1['TOTAL_CUSTOMER_OVERDUE'] / grp2['TOTAL_CUSTOMER_DEBT']

    del grp1['TOTAL_CUSTOMER_OVERDUE']

    app_bureau = app_bureau.merge(grp1, on='SK_ID_CURR', how='left')
    app_bureau['OVERDUE_DEBT_RATIO'] = app_bureau['OVERDUE_DEBT_RATIO'].fillna(0)
    app_bureau['OVERDUE_DEBT_RATIO'] = app_bureau.replace([np.inf, -np.inf], 0)
    app_bureau['OVERDUE_DEBT_RATIO'] = pd.to_numeric(app_bureau['OVERDUE_DEBT_RATIO'], downcast='float')

    app_bureau_test = app_bureau_test.merge(grp1, on='SK_ID_CURR', how='left')
    app_bureau_test['OVERDUE_DEBT_RATIO'] = app_bureau_test['OVERDUE_DEBT_RATIO'].fillna(0)
    app_bureau_test['OVERDUE_DEBT_RATIO'] = app_bureau_test.replace([np.inf, -np.inf], 0)
    app_bureau_test['OVERDUE_DEBT_RATIO'] = pd.to_numeric(app_bureau_test['OVERDUE_DEBT_RATIO'], downcast='float')

    gc.collect()

    # combine previous_application
    grp = previous_application[['SK_ID_CURR', 'SK_ID_PREV']].groupby(by=['SK_ID_CURR'])['SK_ID_PREV']\
        .count().reset_index().rename(columns={'SK_ID_PREV': 'PREV_APP_COUNT'})
    train = app_bureau.merge(grp, on=['SK_ID_CURR'], how='left')
    test = app_bureau_test.merge(grp, on=['SK_ID_CURR'], how='left')

    train['PREV_APP_COUNT'] = train['PREV_APP_COUNT'].fillna(0)
    test['PREV_APP_COUNT'] = test['PREV_APP_COUNT'].fillna(0)

    # numerical features
    grp = previous_application.drop('SK_ID_PREV', axis=1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    grp.columns = ['PREV_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]

    train = train.merge(grp, on=['SK_ID_CURR'], how='left')
    train.update(train[grp.columns].fillna(0))
    test = test.merge(grp, on=['SK_ID_CURR'], how='left')
    test.update(test[grp.columns].fillna(0))

    # categorical features
    prev_categorical = pd.get_dummies(previous_application.select_dtypes('object'))
    prev_categorical['SK_ID_CURR'] = previous_application['SK_ID_CURR']
    grp = prev_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp.columns = ['PREV_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]

    train = train.merge(grp, on=['SK_ID_CURR'], how='left')
    train.update(train[grp.columns].fillna(0))
    test = test.merge(grp, on=['SK_ID_CURR'], how='left')
    test.update(test[grp.columns].fillna(0))

    gc.collect()

    # combine POS_CASH_balance data
    # numerical features
    grp = pos_cash.drop('SK_ID_PREV', axis=1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    grp.columns = ['POS_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]

    train = train.merge(grp, on=['SK_ID_CURR'], how='left')
    train.update(train[grp.columns].fillna(0))
    test = test.merge(grp, on=['SK_ID_CURR'], how='left')
    test.update(test[grp.columns].fillna(0))

    # categorical features
    pos_cash_categorical = pd.get_dummies(pos_cash.select_dtypes('object'))
    pos_cash_categorical['SK_ID_CURR'] = pos_cash['SK_ID_CURR']
    grp = pos_cash_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp.columns = ['POS_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]

    train = train.merge(grp, on=['SK_ID_CURR'], how='left')
    train.update(train[grp.columns].fillna(0))
    test = test.merge(grp, on=['SK_ID_CURR'], how='left')
    test.update(test[grp.columns].fillna(0))

    gc.collect()

    # combine installments_payments data
    # numerical features
    grp = installments_payments.drop('SK_ID_PREV', axis=1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    grp.columns = ['INSTA_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]

    train = train.merge(grp, on=['SK_ID_CURR'], how='left')
    train.update(train[grp.columns].fillna(0))
    test = test.merge(grp, on=['SK_ID_CURR'], how='left')
    test.update(test[grp.columns].fillna(0))

    gc.collect()

    # combine credit_card_balance
    # numerical features
    grp = credit_card_balance.drop('SK_ID_PREV', axis=1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    grp.columns = ['CREDIT_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]

    train = train.merge(grp, on=['SK_ID_CURR'], how='left')
    train.update(train[grp.columns].fillna(0))
    test = test.merge(grp, on=['SK_ID_CURR'], how='left')
    test.update(test[grp.columns].fillna(0))

    # categorical features
    credit_categorical = pd.get_dummies(credit_card_balance.select_dtypes('object'))
    credit_categorical['SK_ID_CURR'] = credit_card_balance['SK_ID_CURR']
    grp = credit_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp.columns = ['CREDIT_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]

    train = train.merge(grp, on=['SK_ID_CURR'], how='left')
    train.update(train[grp.columns].fillna(0))
    test = test.merge(grp, on=['SK_ID_CURR'], how='left')
    test.update(test[grp.columns].fillna(0))

    return train, test




