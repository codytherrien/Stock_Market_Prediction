import numpy as np
import scipy
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import project_functions2 as pf

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

def train_multiple_random_forests(combine_df, split_time, stock_dfs, min_forest_size, max_forest_size, min_features, max_features, forest_step=10):
    X_train, y_train, X_test, y_test = pf.multi_stock_train_test_split(combine_df, split_time, stock_dfs)

    columns = ['Forest Size', 'Features', 'Training Score', 'Test Score']
    df = pd.DataFrame(columns=columns)
    max_score = 0
    max_score_depth = 0
    for num_features in range(min_features, max_features + 1):
        for forest_size in range(min_forest_size, max_forest_size + 1, forest_step):
            clf = RandomForestRegressor(n_estimators=forest_size, max_features=num_features,
                                         random_state=0).fit(X_train, y_train)
            test_score = clf.score(X_test, y_test)
            train_score = clf.score(X_train, y_train)
            df.loc[len(df)] = [forest_size, num_features, train_score, test_score]

    return df

def train_multiple_boosted_DTs(combine_df, split_time, stock_dfs, min_depth, max_depth, min_features, max_features):
    X_train, y_train, X_test, y_test = pf.multi_stock_train_test_split(combine_df, split_time, stock_dfs)

    columns = ['Depth', 'Features', 'Training Score', 'Test Score']
    df = pd.DataFrame(columns=columns)
    max_score = 0
    max_score_depth = 0
    for num_features in range(min_features, max_features + 1):
        for depth in range(min_depth, max_depth + 1):
            gbdt = GradientBoostingRegressor(max_depth=depth, max_features=num_features,
                                         random_state=0).fit(X_train, y_train)
            test_score = gbdt.score(X_test, y_test)
            train_score = gbdt.score(X_train, y_train)
            df.loc[len(df)] = [depth, num_features, train_score, test_score]

    return df