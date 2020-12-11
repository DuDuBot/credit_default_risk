import pandas as pd
from pandas import Series
import pandas.core.algorithms as algos

import re
import traceback

import numpy as np
import scipy.stats.stats as stats


def helper_1(Y, X, n=20):
    df1 = pd.DataFrame({"X": X, "Y": Y})
    flag_1 = df1[['X', 'Y']][df1.X.isnull()]
    flag_2 = df1[['X', 'Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": flag_2.X, "Y": flag_2.Y, "Bucket": pd.qcut(flag_2.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = 3
        bins = algos.quantile(flag_2.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1] - (bins[1] / 2)
        d1 = pd.DataFrame(
            {"X": flag_2.X, "Y": flag_2.Y, "Bucket": pd.cut(flag_2.X, np.unique(bins), include_lowest=True)})
        d2 = d1.groupby('Bucket', as_index=True)

    result = pd.DataFrame({}, index=[])
    result["MIN_VALUE"] = d2.min().X
    result["MAX_VALUE"] = d2.max().X
    result["COUNT"] = d2.count().Y
    result["EVENT"] = d2.sum().Y
    result["NONEVENT"] = d2.count().Y - d2.sum().Y
    result = result.reset_index(drop=True)

    if len(flag_1.index) > 0:
        temp = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
        temp["MAX_VALUE"] = np.nan
        temp["COUNT"] = flag_1.count().Y
        temp["EVENT"] = flag_1.sum().Y
        temp["NONEVENT"] = flag_1.count().Y - flag_1.sum().Y
        result = result.append(temp, ignore_index=True)

    result["EVENT_RATE"] = result.EVENT / result.COUNT
    result["NON_EVENT_RATE"] = result.NONEVENT / result.COUNT
    result["DIST_EVENT"] = result.EVENT / result.sum().EVENT
    result["DIST_NON_EVENT"] = result.NONEVENT / result.sum().NONEVENT
    result["WOE"] = np.log(result.DIST_EVENT / result.DIST_NON_EVENT)
    result["IV"] = (result.DIST_EVENT - result.DIST_NON_EVENT) * np.log(result.DIST_EVENT / result.DIST_NON_EVENT)
    result["VAR_NAME"] = "VAR"
    result = result[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
             'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
    result = result.replace([np.inf, -np.inf], 0)
    result.IV = result.IV.sum()

    return result


def helper_2(Y, X):
    df1 = pd.DataFrame({"X": X, "Y": Y})
    flag_1 = df1[['X', 'Y']][df1.X.isnull()]
    flag_2 = df1[['X', 'Y']][df1.X.notnull()]
    grouped_x = flag_2.groupby('X', as_index=True)

    result = pd.DataFrame({}, index=[])
    result["COUNT"] = grouped_x.count().Y
    result["MIN_VALUE"] = grouped_x.sum().Y.index
    result["MAX_VALUE"] = result["MIN_VALUE"]
    result["EVENT"] = grouped_x.sum().Y
    result["NONEVENT"] = grouped_x.count().Y - grouped_x.sum().Y

    if len(flag_1.index) > 0:
        temp = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
        temp["MAX_VALUE"] = np.nan
        temp["COUNT"] = flag_1.count().Y
        temp["EVENT"] = flag_1.sum().Y
        temp["NONEVENT"] = flag_1.count().Y - flag_1.sum().Y
        result = result.append(temp, ignore_index=True)

    result["EVENT_RATE"] = result.EVENT / result.COUNT
    result["NON_EVENT_RATE"] = result.NONEVENT / result.COUNT
    result["DIST_EVENT"] = result.EVENT / result.sum().EVENT
    result["DIST_NON_EVENT"] = result.NONEVENT / result.sum().NONEVENT
    result["WOE"] = np.log(result.DIST_EVENT / result.DIST_NON_EVENT)
    result["IV"] = (result.DIST_EVENT - result.DIST_NON_EVENT) * np.log(result.DIST_EVENT / result.DIST_NON_EVENT)
    result["VAR_NAME"] = "VAR"
    result = result[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
             'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
    result = result.replace([np.inf, -np.inf], 0)
    result.IV = result.IV.sum()
    result = result.reset_index(drop=True)

    return result


def iv_calculator(dataframe, target):
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]

    count = -1
    for i in dataframe.dtypes.index:
        if i.upper() not in (final.upper()):
            if np.issubdtype(dataframe[i], np.number) and len(Series.unique(dataframe[i])) > 2:
                temp = helper_1(target, dataframe[i])
                temp["VAR_NAME"] = i
                count += 1
            else:
                temp = helper_2(target, dataframe[i])
                temp["VAR_NAME"] = i
                count += 1

            if count == 0:
                iv_dataframe = temp
            else:
                iv_dataframe = iv_dataframe.append(temp, ignore_index=True)

    iv = pd.DataFrame({'IV': iv_dataframe.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return iv_dataframe, iv

