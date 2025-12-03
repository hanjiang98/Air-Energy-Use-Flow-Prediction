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
from xgboost import XGBClassifier, XGBRegressor
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


# 在同一年的数据上训练和测试模型，评估模型对当前数据的预测能力。
def simultaneous_test(df_se, df_de, features, best_params, best_params_reg, save=True, name=None):
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

        model_reg = XGBRegressor(**best_params_reg)
        model_reg.fit(X_train, y_reg_train)
        y_reg_pred = model_reg.predict(X_test)

        ros = RandomUnderSampler()    # 处理数据的类不平衡问题
        X_train, y_train = ros.fit_resample(X_train, y_train)
        y_train_null = y_train.copy()
        np.random.shuffle(y_train_null)
        model = XGBClassifier(**best_params)  # 作为预测模型
        # model = XGBClassifier(objective='multi:softmax', num_class=3, use_label_encoder=False)
        model.fit(X_train, y_train)
        model_null = XGBClassifier(**best_params)
        model_null.fit(X_train, y_train_null)
        y_pred = model.predict(X_test)
        y_pred_null = model_null.predict(X_test)

        res_df_de.loc[res_df_de.Year == year, 'simultaneous_pred'] = y_pred
        res_df_de.loc[res_df_de.Year == year, 'simultaneous_null'] = y_pred_null
        res_df_de.loc[res_df_de.Year == year, 'reg_pred'] = y_reg_pred
    if save:
        res_df_de.to_csv('./results/'+name+'.csv')
    return res_df_de


# 在不同年份的数据上训练和测试，评估模型对未来数据的预测能力。
def nonsimultaneous_test(df_train, df_test, features, best_params, best_params_reg, save=True, name=None):
    if name is None:
        name = ''.join([w[0] for w in features]) + '_nonsimultaneous'
    else:
        name = name + '_nonsimultaneous'
    year_list = list(df_test.Year.unique())
    preds = []
    for year_train in tqdm(year_list):
        X_train, y_train, y_reg_train = df_to_XY(df_train[df_train.Year == year_train], features)

        model_reg = XGBRegressor(**best_params_reg)
        model_reg.fit(X_train, y_reg_train)
        i = year_list.index(year_train)
        for year_test in year_list[i:]:
            X_test, y_test, y_reg_test = df_to_XY(df_test[df_test.Year == year_test], features)
            y_reg_pred = model_reg.predict(X_test)
            preds.append([year_train, year_test, y_reg_test, y_reg_pred])

        ros = RandomUnderSampler()
        X_train, y_train = ros.fit_resample(X_train, y_train)
        y_train_null = y_train.copy()
        np.random.shuffle(y_train_null)
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        model_null = XGBClassifier(**best_params)
        model_null.fit(X_train, y_train_null)
        # i = year_list.index(year_train)
        for year_test in year_list[i:]:
            X_test, y_test, y_reg_test = df_to_XY(df_test[df_test.Year == year_test], features)
            y_pred = model.predict(X_test)
            y_null = model_null.predict(X_test)
            preds.append([year_test, y_test, y_pred, y_null])
    if save:
        import pickle
        with open('./results/' + name +'.pkl', 'wb') as f:
            pickle.dump(preds, f)
    return preds


def all_shap_values(df1, df2, features, best_params, best_params_reg, save=True, name=None):    # SHAP计算
    import shap    # 一个库，用于解释机器学习模型的预测
    if name is None:
        name = ''.join([w[0] for w in features]) + '_SHAP'
    else:
        name = name + '_SHAP'

    def get_temporal_order(shap_list):
        importance_array = []
        for shap_values in shap_list:
            array = -np.abs(shap_values).mean(axis=0)
            ranks = ss.rankdata(array)
            importance_array.append(ranks)
        return np.array(importance_array)

    shap_values_list = []
    shap_values_reg_list = []
    test_list = []
    year_list = []
    for i in tqdm(df2.Year.unique()):
        X_train, y_train, y_reg_train = df_to_XY(df1[df1.Year == i].drop(columns=['Year']), features)
        X_test, y_test, y_reg_test = df_to_XY(df2[df2.Year == i].drop(columns=['Year']), features)

        model_reg = XGBRegressor(**best_params_reg)
        model_reg.fit(X_train, y_reg_train)
        explainer_reg = shap.TreeExplainer(model_reg)
        shap_values_reg = explainer_reg.shap_values(X_test, check_additivity=False)

        ros = RandomUnderSampler()
        X_train, y_train = ros.fit_resample(X_train, y_train)
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test, check_additivity=False)

        year_list.append(i)
        test_list.append(pd.DataFrame(X_test, columns=features))
        shap_values_list.append(shap_values)
        shap_values_reg_list.append(shap_values_reg)
    if save:
        import pickle
        with open('./results/'+name+'.pkl', 'wb') as f:
            pickle.dump((test_list, year_list, shap_values_list, shap_values_reg_list), f)
    return test_list, year_list, shap_values_list, shap_values_reg_list

def all_models_for_pdp(df1, df2, features, best_params, best_params_reg, save=True, name=None):
    if name is None:
        name = ''.join([w[0] for w in features]) + '_PDP'
    else:
        name = name + '_PDP'

    model_reg_list = []
    X_test_list = []
    year_list = []
    model_list = []

    for year in tqdm(df2.Year.unique()):
        # 构造训练集和测试集
        X_train, y_train, y_reg_train = df_to_XY(df1[df1.Year == year].drop(columns=['Year']), features)
        X_test, y_test, y_reg_test = df_to_XY(df2[df2.Year == year].drop(columns=['Year']), features)

        # 回归模型
        model_reg = XGBRegressor(**best_params_reg)
        model_reg.fit(X_train, y_reg_train)

        # 分类模型
        ros = RandomUnderSampler()
        X_train, y_train = ros.fit_resample(X_train, y_train)
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        # 保存模型和测试集
        year_list.append(year)
        model_reg_list.append(model_reg)
        model_list.append(model)
        X_test_list.append(pd.DataFrame(X_test, columns=features))
    if save:
        import pickle
        with open('./results/'+name+'.pkl', 'wb') as f:
            pickle.dump((X_test_list, year_list, model_list, model_reg_list), f)

    return X_test_list, year_list, model_list, model_reg_list

# 对应论文prediction部分
# 设置了11个函数来处理和分类网络数据，用于预测和解释网络中的边（连接）的行为。
def BTF(train, test):  # unweighted Topological Features
    name = 'Air_Classification_BTF'
    features = ['Common Neighbor', 'Salton', 'Jaccard', 'Sorensen', 'Hub Promoted',
               'Hub Depressed', 'Leicht Holme Newman', 'Preferential Attachment',
               'Adamic Adar', 'Resource Allocation', 'Local Path']
    all_shap_values(train, test, features, best_params, best_params_reg, name=name)
    simultaneous_test(train, test, features, best_params, best_params_reg, name=name)
    nonsimultaneous_test(train, test, features, best_params, best_params_reg, name=name)
    all_models_for_pdp(train, test, features, best_params, best_params_reg, name=name)

def WTF(train, test):  # Weighted Topological Features
    name = 'Air_Classification_WTF'
    features = ['Weighted Common Neighbor', 'Weighted Salton', 'Weighted Preferential Attachment',
    'Weighted Leicht Holme Newman', 'Weighted Sorensen', 'Weighted Hub Promoted', 'Weighted Hub Depressed', 'Weighted Local Path', 'Weighted Resource Allocation',
    'Weighted Adamic Adar', 'Weighted Jaccard']
    for c in data.columns:
        if 'Weighted' in c:
            features.append(c)
    simultaneous_test(train, test, features, best_params, best_params_reg, name=name)
    nonsimultaneous_test(train, test, features, best_params, best_params_reg, name=name)
    all_shap_values(train, test, features, best_params, best_params_reg, name=name)
    all_models_for_pdp(train, test, features, best_params, best_params_reg, name=name)

def WWW(train, test):    # edge weights
    name = 'Air_Classification_WWW'
    features = ['Curr FWeight']
    simultaneous_test(train, test, features, best_params, best_params_reg, name = name)
    nonsimultaneous_test(train, test, features, best_params, best_params_reg, name = name)
    all_shap_values(train, test, features, best_params, best_params_reg, name = name)
    all_models_for_pdp(train, test, features, best_params, best_params_reg, name=name)

def BTFW(train, test):      # Basic Topological and Weighted Features
    name = 'Air_Classification_BTFW'
    features = ['Common Neighbor', 'Salton', 'Jaccard', 'Sorensen', 'Hub Promoted',
               'Hub Depressed', 'Leicht Holme Newman', 'Preferential Attachment',
               'Adamic Adar', 'Resource Allocation', 'Local Path', 'Curr FWeight']

    simultaneous_test(train, test, features, best_params, best_params_reg, name=name)
    nonsimultaneous_test(train, test, features, best_params, best_params_reg, name=name)
    all_shap_values(train, test, features, best_params, best_params_reg, name=name)
    all_models_for_pdp(train, test, features, best_params, best_params_reg, name=name)

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

    global best_params, best_params_reg
    # best_params_reg = {'lambda': 0.5650701862593042, 'alpha': 0.0016650896783581535,
    #                'colsample_bytree': 1.0, 'subsample': 0.5, 'learning_rate': 0.009,
    #                'n_estimators': 625, 'objective': 'reg:squarederror', 'max_depth': 5, 'min_child_weight': 6}
    #
    # best_params = {'lambda': 0.5650701862593042, 'alpha': 0.0016650896783581535,
    #                'colsample_bytree': 1.0, 'subsample': 0.5, 'learning_rate': 0.009,
    #                'n_estimators': 625, 'objective': 'multi:softmax', 'max_depth': 5, 'min_child_weight': 6,
    #                'num_class': 3}  # 添加类别总数为3
    best_params_reg = load_best_params('best_xgb_reg_params.json')

    best_params = load_best_params('best_xgb_params.json')

    # best_params = None
    for data_path in ['data/features/cnair_2014_2024_2.csv']:
        data = pd.read_csv(data_path)
        # print(data.columns)
        train, test = get_edge_slice(data)
        BTF(train, test)
        BTFW(train, test)
        WTF(train, test)
        WWW(train, test)

