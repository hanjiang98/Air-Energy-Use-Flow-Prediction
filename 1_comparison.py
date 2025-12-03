import sys
import random
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import scipy.stats as ss


from datetime import date
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import confusion_matrix,balanced_accuracy_score, mean_squared_error,r2_score,mean_absolute_error

# sys.path.append('src')
from src.plot_style import *


def get_edge_slice(data, f_train_e=0.7, seed=30):  # 随机选择70%作为训练集，其余为测试集
    data = data.drop('Unnamed: 0', axis=1)
    df = data
    edges = set(df.Edge.unique())
    random.seed(seed)
    edge_train = set(random.sample(list(edges), int(f_train_e*len(edges))))
    edge_test = set([e for e in edges if e not in edge_train])
    df_se = df.loc[df['Edge'].isin(edge_train)].drop(columns=['Edge'])
    df_de = df.loc[df['Edge'].isin(edge_test)].drop(columns=['Edge'])
    return df_se, df_de


def df_to_XY(df, features, target='Dynamic'):
    if 'Year' in df.columns:
        df = df.drop(columns=['Year'])
    if "Edge" in df.columns:
        df = df.drop(columns=['Edge'])
    X = df.loc[:, features].to_numpy()
    y = df.loc[:, df.columns == target].to_numpy()
    return X, y, df.loc[:, df.columns == 'Next Weight'].to_numpy()


# # 在同一年的数据上训练和测试模型，评估模型对当前数据的预测能力。
# def simultaneous_test(df_se, df_de, features, best_params, best_params_reg, save=True, name=None):
#     if name is None:
#         name = ''.join([w[0] for w in features]) + '_simultaneous'
#     else:
#         name = name + '_simultaneous'
#     year_list = list(df_se.Year.unique())
#     res_df_de = df_de.copy()
#     res_df_de['simultaneous_pred'] = np.nan
#     res_df_de['simultaneous_null'] = np.nan
#     for year in tqdm(year_list):
#         X_train, y_train, y_reg_train = df_to_XY(df_se[df_se.Year == year], features)
#         X_test, y_test, y_reg_test = df_to_XY(df_de[df_de.Year == year], features)
#
#         model_reg = XGBRegressor(**best_params_reg)
#         model_reg.fit(X_train, y_reg_train)
#         y_reg_pred = model_reg.predict(X_test)
#
#         ros = RandomUnderSampler()    # 处理数据的类不平衡问题
#         X_train, y_train = ros.fit_resample(X_train, y_train)
#         y_train_null = y_train.copy()
#         np.random.shuffle(y_train_null)
#         model = XGBClassifier(**best_params)  # 作为预测模型
#         # model = XGBClassifier(objective='multi:softmax', num_class=3, use_label_encoder=False)
#         model.fit(X_train, y_train)
#         model_null = XGBClassifier(**best_params)
#         model_null.fit(X_train, y_train_null)
#         y_pred = model.predict(X_test)
#         y_pred_null = model_null.predict(X_test)
#
#         res_df_de.loc[res_df_de.Year == year, 'simultaneous_pred'] = y_pred
#         res_df_de.loc[res_df_de.Year == year, 'simultaneous_null'] = y_pred_null
#         res_df_de.loc[res_df_de.Year == year, 'reg_pred'] = y_reg_pred
#     if save:
#         res_df_de.to_csv('./results/'+name+'.csv')
#     return res_df_de

def clf(df_se, df_de, features, Model, best_params, save=True, name=None):
    if name is None:
        name = ''.join([w[0] for w in features]) + '_simultaneous'
    else:
        name = name + '_simultaneous'
    year_list = list(df_se.Year.unique())
    res_df_de = df_de.copy()
    res_df_de['simultaneous_pred'] = np.nan
    res_df_de['simultaneous_null'] = np.nan
    for year in tqdm(year_list):
        X_train, y_train, y_reg_train = df_to_XY(df_se[df_se.Year == year], features)
        X_test, y_test, y_reg_test = df_to_XY(df_de[df_de.Year == year], features)

        ros = RandomUnderSampler()    # 处理数据的类不平衡问题
        X_train, y_train = ros.fit_resample(X_train, y_train)
        y_train_null = y_train.copy()
        np.random.shuffle(y_train_null)
        model = Model(**best_params)  # 作为预测模型
        # model = Model(objective='multi:softmax', num_class=3, use_label_encoder=False)
        model.fit(X_train, y_train)
        model_null = Model(**best_params)
        model_null.fit(X_train, y_train_null)
        y_pred = model.predict(X_test)
        y_pred_null = model_null.predict(X_test)

        res_df_de.loc[res_df_de.Year == year, 'simultaneous_pred'] = y_pred
        res_df_de.loc[res_df_de.Year == year, 'simultaneous_null'] = y_pred_null
    if save:
        res_df_de.to_csv('./results/'+name+'.csv')
    return res_df_de

def reg(df_se, df_de, features, Model, best_params_reg, save=True, name=None):
    if name is None:
        name = ''.join([w[0] for w in features]) + '_simultaneous'
    else:
        name = name + '_simultaneous'
    year_list = list(df_se.Year.unique())
    res_df_de = df_de.copy()
    res_df_de['simultaneous_pred'] = np.nan
    res_df_de['simultaneous_null'] = np.nan
    for year in tqdm(year_list):
        X_train, y_train, y_reg_train = df_to_XY(df_se[df_se.Year == year], features)
        X_test, y_test, y_reg_test = df_to_XY(df_de[df_de.Year == year], features)

        model_reg = Model(**best_params_reg)
        model_reg.fit(X_train, y_reg_train)
        y_reg_pred = model_reg.predict(X_test)

        res_df_de.loc[res_df_de.Year == year, 'reg_pred'] = y_reg_pred
    if save:
        res_df_de.to_csv('./results/'+name+'.csv')
    return res_df_de

def load_best_params(json_path):
    """
    加载保存的最优超参数 JSON 文件

    参数:
        json_path: str，超参数 JSON 文件路径

    返回:
        dict，包含最优超参数的字典
    """
    import json
    import os

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到文件: {json_path}")

    with open(json_path, 'r') as f:
        best_params = json.load(f)

    print(f"✅ 成功加载超参数文件: {json_path}")
    return best_params

if __name__ == "__main__":
    # to run classification on bus change the data
    # data =

    # global xgb_best_params, logistic_regression_best_params, rf_best_params, gb_best_params, knn_best_params, mlp_best_params, dt_best_params, lsvc_best_params, hgb_best_params, \
    #     ridge_best_params, lasso_best_params, rf_reg_best_params, gb_reg_best_params, knn_reg_best_params, mlp_reg_best_params, huber_best_params, hgb_reg_best_params, xgb_reg_best_params
    # 加载 XGBRegressor 最优参数
    xgb_best_params = load_best_params('best_xgb_params.json')
    logistic_regression_best_params = load_best_params('best_logistic_params.json')
    rf_best_params = load_best_params('best_rf_params.json')
    gb_best_params = load_best_params('best_gb_params.json')
    knn_best_params = load_best_params('best_knn_params.json')
    mlp_best_params = load_best_params('best_mlp_params.json')
    gnb_best_params = load_best_params('best_gnb_params.json')
    dt_best_params = load_best_params('best_dt_params.json')
    lsvc_best_params = load_best_params('best_lsvc_params.json')
    hgb_best_params = load_best_params('best_hgb_params.json')
    # svc_best_params = load_best_params('best_svc_params.json')
    svc_best_params = {}
    best_params = [xgb_best_params, logistic_regression_best_params, svc_best_params, rf_best_params, gb_best_params,
                   knn_best_params, mlp_best_params, gnb_best_params, dt_best_params, lsvc_best_params, hgb_best_params]

    lr_best_params = load_best_params('best_lr_params.json')
    ridge_best_params = load_best_params('best_ridge_params.json')
    lasso_best_params = load_best_params('best_lasso_params.json')
    rf_reg_best_params = load_best_params('best_rf_reg_params.json')
    # svr_best_params = load_best_params('best_svr_params.json')
    svr_best_params = {}
    gb_reg_best_params = load_best_params('best_gb_reg_params.json')
    knn_reg_best_params = load_best_params('best_knn_reg_params.json')
    mlp_reg_best_params = load_best_params('best_mlp_reg_params.json')
    huber_best_params = load_best_params('best_huber_params.json')
    hgb_reg_best_params = load_best_params('best_hgb_reg_params.json')
    xgb_reg_best_params = load_best_params('best_xgb_reg_params.json')
    best_params_reg = [xgb_reg_best_params, lr_best_params, ridge_best_params, lasso_best_params, svr_best_params, rf_reg_best_params,
                       gb_reg_best_params, knn_reg_best_params, mlp_reg_best_params, huber_best_params, hgb_reg_best_params, ]

    # best_params = None
    for data_path in ['data/features/cnair_2014_2024_2.csv']:
        data = pd.read_csv(data_path)
        # print(data.columns)
        train, test = get_edge_slice(data)

        features = ['Common Neighbor', 'Salton', 'Jaccard', 'Sorensen', 'Hub Promoted',
                    'Hub Depressed', 'Leicht Holme Newman', 'Preferential Attachment',
                    'Adamic Adar', 'Resource Allocation', 'Local Path', 'Curr FWeight']

        from xgboost import XGBClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import HistGradientBoostingClassifier

        from xgboost import XGBRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import Ridge
        from sklearn.linear_model import Lasso
        from sklearn.svm import SVR
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.linear_model import HuberRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor

        Models = [XGBClassifier, LogisticRegression, SVC, RandomForestClassifier, GradientBoostingClassifier,
                  KNeighborsClassifier, MLPClassifier, GaussianNB, DecisionTreeClassifier, LinearSVC, HistGradientBoostingClassifier]

        Models_reg = [XGBRegressor, LinearRegression, Ridge, Lasso, SVR, RandomForestRegressor, GradientBoostingRegressor,
                      KNeighborsRegressor, MLPRegressor, HuberRegressor, HistGradientBoostingRegressor]

        names = ['xgb_clf', 'logistic_clf', 'svc_clf', 'rf_clf', 'gb_clf',
                 'knn_clf', 'mlp_clf', 'gnb_clf', 'dt_clf', 'lsvc_clf', 'hgb_clf']

        names_reg = ['xgb_reg', 'lr_reg', 'ridge_reg', 'lasso_reg', 'svr_reg',
                     'rf_reg', 'gb_reg', 'knn_reg', 'mlp_reg', 'huber_reg', 'hgb_reg']

        # for item in range(len(Models)):
        #     print(f"Now is {names[item]}:")
        #     clf(train, test, features, Models[item], best_params[item], name=f'Air_Classification_{names[item]}')

        for item in range(len(Models_reg)):
            print(f"Now is {names_reg[item]}:")
            reg(train, test, features, Models_reg[item], best_params_reg[item], name=f'Air_Classification_{names_reg[item]}')


