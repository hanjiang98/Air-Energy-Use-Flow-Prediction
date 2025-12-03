# -*- coding: utf-8 -*-
"""
# @Time    : 2025/10/24
# @Author  : Hanjiang Dong
# @Intro   : 
"""

import sys
import random
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import scipy.stats as ss


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
import json


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

# ------------分类-----------

def search_xgb_best_params(df_se, features, save_path='best_xgb_params.json'):
    """
        用最后一年的数据对 XGBClassifier 进行超参数搜索，并保存最优参数。

        参数：
            df_se: DataFrame，训练数据集，需包含 'Year'
            features: list[str]，用于训练的特征名列表
            save_path: str，保存最优参数的 JSON 文件路径
        """
    # 1. 获取最后一年数据
    last_year = df_se.Year.max()
    df_last = df_se[df_se.Year == last_year]

    # 2. 提取特征和标签
    X_train, y_train, _ = df_to_XY(df_last, features)

    # 3. 处理类别不平衡
    ros = RandomUnderSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    # 4. 定义超参数搜索空间
    param_dist = {
        'n_estimators': [100, 300, 500, 700],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
        'subsample': [0.5, 0.7, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 6],
        'reg_alpha': [0.0, 0.001, 0.01, 0.1],
        'reg_lambda': [0.0, 0.1, 0.5, 1.0],
        'objective': ['multi:softmax'],
        'num_class': [3],
        'use_label_encoder': [False],
        'eval_metric': ['mlogloss']
    }

    # 5. 执行随机搜索
    print(f"🔍 开始使用 {last_year} 年数据进行超参数搜索...")
    xgb = XGBClassifier()
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=30,
        scoring='accuracy',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    # 6. 获取最优参数
    best_params = search.best_params_
    print("\n✅ 最优参数如下：")
    print(best_params)

    # 7. 保存最优参数到 JSON 文件
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
        print(f"\n📁 最优参数已保存到 {save_path}")

    return best_params

def search_logistic_regression_best_params(df_se, features, save_path='best_logistic_params.json'):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X_train, y_train, _ = df_to_XY(df_se[df_se.Year == last_year], features)

    from imblearn.under_sampling import RandomUnderSampler
    ros = RandomUnderSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)

    param_dist = {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'C': np.logspace(-4, 4, 20),
        'solver': ['saga', 'liblinear'],
        'max_iter': [100, 200, 500]
    }

    model = LogisticRegression(multi_class='auto')
    search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=1)
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print("✅ LogisticRegression 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_svc_best_params(df_se, features, save_path='best_svc_params.json'):
    from sklearn.svm import SVC
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X_train, y_train, _ = df_to_XY(df_se[df_se.Year == last_year], features)

    from imblearn.under_sampling import RandomUnderSampler
    ros = RandomUnderSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)

    param_dist = {
        'C': np.logspace(-3, 2, 10),
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    model = SVC()
    search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=1)
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print("✅ SVC 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_rf_best_params(df_se, features, save_path='best_rf_params.json'):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X_train, y_train, _ = df_to_XY(df_se[df_se.Year == last_year], features)
    from imblearn.under_sampling import RandomUnderSampler
    ros = RandomUnderSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)

    param_dist = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    model = RandomForestClassifier()
    search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=1)
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print("✅ RandomForest 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_gb_best_params(df_se, features, save_path='best_gb_params.json'):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X_train, y_train, _ = df_to_XY(df_se[df_se.Year == last_year], features)
    from imblearn.under_sampling import RandomUnderSampler
    ros = RandomUnderSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)

    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
        'subsample': [0.6, 0.8, 1.0]
    }

    model = GradientBoostingClassifier()
    search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=1)
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print("✅ GradientBoosting 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_knn_best_params(df_se, features, save_path='best_knn_params.json'):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X_train, y_train, _ = df_to_XY(df_se[df_se.Year == last_year], features)
    from imblearn.under_sampling import RandomUnderSampler
    ros = RandomUnderSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)

    param_dist = {
        'n_neighbors': list(range(3, 21)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    model = KNeighborsClassifier()
    search = RandomizedSearchCV(model, param_dist, n_iter=15, cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=1)
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print("✅ KNN 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_mlp_best_params(df_se, features, save_path='best_mlp_params.json'):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X_train, y_train, _ = df_to_XY(df_se[df_se.Year == last_year], features)
    from imblearn.under_sampling import RandomUnderSampler
    ros = RandomUnderSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)

    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 100, 50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': [1e-5, 1e-4, 1e-3],
        'learning_rate': ['constant', 'adaptive']
    }

    model = MLPClassifier(max_iter=300)
    search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=1)
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print("✅ MLP 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_gnb_best_params(df_se, features, save_path='best_gnb_params.json'):
    from sklearn.naive_bayes import GaussianNB
    import json

    last_year = df_se.Year.max()
    X_train, y_train, _ = df_to_XY(df_se[df_se.Year == last_year], features)

    # 无超参数可调，只保存默认参数
    best_params = {'var_smoothing': 1e-9}
    print("✅ GaussianNB 无需搜索，使用默认参数:", best_params)

    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_dt_best_params(df_se, features, save_path='best_dt_params.json'):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X_train, y_train, _ = df_to_XY(df_se[df_se.Year == last_year], features)
    from imblearn.under_sampling import RandomUnderSampler
    ros = RandomUnderSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)

    param_dist = {
        'max_depth': [None, 3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    model = DecisionTreeClassifier()
    search = RandomizedSearchCV(model, param_dist, n_iter=15, cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=1)
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print("✅ DecisionTree 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_lsvc_best_params(df_se, features, save_path='best_lsvc_params.json'):
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X_train, y_train, _ = df_to_XY(df_se[df_se.Year == last_year], features)
    from imblearn.under_sampling import RandomUnderSampler
    ros = RandomUnderSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)

    param_dist = {
        'C': np.logspace(-3, 2, 10),
        'penalty': ['l2'],
        'loss': ['squared_hinge'],
        'max_iter': [1000, 2000]
    }

    model = LinearSVC()
    search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=1)
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print("✅ LinearSVC 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_hgb_best_params(df_se, features, save_path='best_hgb_params.json'):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X_train, y_train, _ = df_to_XY(df_se[df_se.Year == last_year], features)
    from imblearn.under_sampling import RandomUnderSampler
    ros = RandomUnderSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)

    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_iter': [100, 200, 300],
        'max_leaf_nodes': [15, 31, 63],
        'max_depth': [None, 5, 10],
        'l2_regularization': [0.0, 0.1, 1.0],
        'early_stopping': [True]
    }

    model = HistGradientBoostingClassifier()
    search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=1)
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print("✅ HistGBClassifier 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

# ------------回归-----------
def search_lr_best_params(df_se, features, save_path='best_lr_params.json'):
    import json
    best_params = {}  # 无需调参
    print("✅ LinearRegression 无需调参")
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_ridge_best_params(df_se, features, save_path='best_ridge_params.json'):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X, _, y = df_to_XY(df_se[df_se.Year == last_year], features)

    param_dist = {
        'alpha': np.logspace(-4, 4, 50),
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
    }

    model = Ridge()
    search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring='r2', random_state=42, n_jobs=-1, verbose=1)
    search.fit(X, y)

    best_params = search.best_params_
    print("✅ Ridge 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_lasso_best_params(df_se, features, save_path='best_lasso_params.json'):
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X, _, y = df_to_XY(df_se[df_se.Year == last_year], features)

    param_dist = {
        'alpha': np.logspace(-4, 1, 50),
        'max_iter': [1000, 2000, 5000]
    }

    model = Lasso()
    search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring='r2', random_state=42, n_jobs=-1, verbose=1)
    search.fit(X, y)

    best_params = search.best_params_
    print("✅ Lasso 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_svr_best_params(df_se, features, save_path='best_svr_params.json'):
    from sklearn.svm import SVR
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X, _, y = df_to_XY(df_se[df_se.Year == last_year], features)

    param_dist = {
        'C': np.logspace(-2, 2, 10),
        'epsilon': [0.01, 0.1, 0.2],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }

    model = SVR()
    search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring='r2', random_state=42, n_jobs=-1, verbose=1)
    search.fit(X, y)

    best_params = search.best_params_
    print("✅ SVR 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_rf_reg_best_params(df_se, features, save_path='best_rf_reg_params.json'):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X, _, y = df_to_XY(df_se[df_se.Year == last_year], features)

    param_dist = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    model = RandomForestRegressor()
    search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring='r2', random_state=42, n_jobs=-1, verbose=1)
    search.fit(X, y)

    best_params = search.best_params_
    print("✅ RandomForestRegressor 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_gb_reg_best_params(df_se, features, save_path='best_gb_reg_params.json'):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X, _, y = df_to_XY(df_se[df_se.Year == last_year], features)

    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
        'subsample': [0.6, 0.8, 1.0]
    }

    model = GradientBoostingRegressor()
    search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring='r2', random_state=42, n_jobs=-1, verbose=1)
    search.fit(X, y)

    best_params = search.best_params_
    print("✅ GradientBoostingRegressor 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_knn_reg_best_params(df_se, features, save_path='best_knn_reg_params.json'):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X, _, y = df_to_XY(df_se[df_se.Year == last_year], features)

    param_dist = {
        'n_neighbors': list(range(3, 20)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    model = KNeighborsRegressor()
    search = RandomizedSearchCV(model, param_dist, n_iter=15, cv=3, scoring='r2', random_state=42, n_jobs=-1, verbose=1)
    search.fit(X, y)

    best_params = search.best_params_
    print("✅ KNN Regressor 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_mlp_reg_best_params(df_se, features, save_path='best_mlp_reg_params.json'):
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X, _, y = df_to_XY(df_se[df_se.Year == last_year], features)

    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [1e-5, 1e-4, 1e-3],
        'learning_rate': ['constant', 'adaptive']
    }

    model = MLPRegressor(max_iter=300)
    search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring='r2', random_state=42, n_jobs=-1, verbose=1)
    search.fit(X, y)

    best_params = search.best_params_
    print("✅ MLPRegressor 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_huber_best_params(df_se, features, save_path='best_huber_params.json'):
    from sklearn.linear_model import HuberRegressor
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X, _, y = df_to_XY(df_se[df_se.Year == last_year], features)

    param_dist = {
        'epsilon': [1.1, 1.35, 1.5, 1.75, 2.0],
        'alpha': np.logspace(-4, 0, 10)
    }

    model = HuberRegressor(max_iter=1000)
    search = RandomizedSearchCV(model, param_dist, n_iter=15, cv=3, scoring='r2', random_state=42, n_jobs=-1, verbose=1)
    search.fit(X, y)

    best_params = search.best_params_
    print("✅ HuberRegressor 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_hgb_reg_best_params(df_se, features, save_path='best_hgb_reg_params.json'):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import RandomizedSearchCV
    import json

    last_year = df_se.Year.max()
    X, _, y = df_to_XY(df_se[df_se.Year == last_year], features)

    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_iter': [100, 200, 300],
        'max_leaf_nodes': [15, 31, 63],
        'max_depth': [None, 5, 10],
        'l2_regularization': [0.0, 0.1, 1.0],
        'early_stopping': [True]
    }

    model = HistGradientBoostingRegressor()
    search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring='r2', random_state=42, n_jobs=-1, verbose=1)
    search.fit(X, y)

    best_params = search.best_params_
    print("✅ HistGradientBoostingRegressor 最优参数:", best_params)
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params

def search_xgb_reg_best_params(df_se, features, save_path='best_xgb_reg_params.json'):
    """
    使用最后一年数据对 XGBRegressor 进行超参数搜索，并保存最优参数。

    参数:
        df_se: DataFrame，包含数据和 'Year' 列
        features: list[str]，特征列名
        save_path: str，保存最优参数的 JSON 文件路径

    返回:
        dict，最优参数字典
    """
    import json
    import numpy as np
    from xgboost import XGBRegressor
    from sklearn.model_selection import RandomizedSearchCV

    # 1. 获取最后一年数据
    last_year = df_se.Year.max()
    df_last = df_se[df_se.Year == last_year]

    # 2. 提取特征和标签
    X_train, _, y_train = df_to_XY(df_last, features)

    # 3. 定义超参数搜索空间
    param_dist = {
        'n_estimators': [100, 300, 500, 700],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
        'subsample': [0.5, 0.7, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 6],
        'reg_alpha': [0.0, 0.001, 0.01, 0.1],
        'reg_lambda': [0.0, 0.1, 0.5, 1.0],
        'gamma': [0, 0.1, 0.3, 0.5]
    }

    # 4. 启动搜索
    print(f"🔍 开始使用 {last_year} 年数据进行 XGBRegressor 超参数搜索...")
    model = XGBRegressor(objective='reg:squarederror', random_state=42)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=30,
        scoring='r2',
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    # 5. 输出并保存结果
    best_params = search.best_params_
    print("✅ XGBRegressor 最优参数:")
    print(best_params)

    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
        print(f"📁 最优参数已保存到 {save_path}")

    return best_params

if __name__ == "__main__":
    # to run classification on bus change the data
    # data =

    # best_params = None
    for data_path in ['data/features/cnair_2014_2024_2.csv']:
        data = pd.read_csv(data_path)
        # print(data.columns)
        train, test = get_edge_slice(data)

        features = ['Common Neighbor', 'Salton', 'Jaccard', 'Sorensen', 'Hub Promoted',
                    'Hub Depressed', 'Leicht Holme Newman', 'Preferential Attachment',
                    'Adamic Adar', 'Resource Allocation', 'Local Path', 'Curr FWeight']

        search_xgb_best_params(train, features)
        search_logistic_regression_best_params(train, features)
        search_rf_best_params(train, features)
        search_gb_best_params(train, features)
        search_knn_best_params(train, features)
        search_mlp_best_params(train, features)
        search_gnb_best_params(train, features)
        search_dt_best_params(train, features)
        search_lsvc_best_params(train, features)
        search_hgb_best_params(train, features)
        # search_svc_best_params(train, features)  # ？？

        search_lr_best_params(train, features)
        search_ridge_best_params(train, features)
        search_lasso_best_params(train, features)
        search_rf_reg_best_params(train, features)
        search_gb_reg_best_params(train, features)
        search_knn_reg_best_params(train, features)
        search_mlp_reg_best_params(train, features)
        search_huber_best_params(train, features)
        search_hgb_reg_best_params(train, features)
        search_xgb_reg_best_params(train, features)
        # search_svr_best_params(train, features)





