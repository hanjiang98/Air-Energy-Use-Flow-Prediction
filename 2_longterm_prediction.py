import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import random
import math
import datetime

from datetime import date
from collections import defaultdict
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, \
mean_squared_error, r2_score, mean_absolute_error

import sys
# sys.path.append("src/")
from src.plot_style import *


def is_edge_in_hedges(edge, hedges):
    # Check both possible directions for the edge
    # 假设H是一个图的字典，H[u][v]['weight']是边(u, v)的权重
    # 假设Hedges是图H中所有边的集合
    # e是一个边的元组，如('fuo', 'kwl')
    # X['Next Weight']是一个列表，你需要向其中添加权重或0
    # (edge[1], edge[0]) in hedges: 这部分首先将 edge 元组的元素反转，即如果 edge 是 (u, v)，则反转后变为 (v, u)。
    # 之后，检查这个反转后的元组是否存在于 hedges 中。
    return edge in hedges or (edge[1], edge[0]) in hedges


def get_feature_vector_add(graph):
    def local_path(G, nodeList, epsilon=0.01):
        A = nx.adjacency_matrix(G, nodelist=nodeList, weight=None).todense()
        return A ** 2 + epsilon * A ** 3

    def l3_path(G, nodeList):
        A = nx.adjacency_matrix(G, nodelist=nodeList, weight=None).todense()
        return A ** 3

    def weighted_local_path(G, nodeList, epsilon=0.01):
        A = nx.adjacency_matrix(G, nodelist=nodeList, weight='weight').todense()
        return A ** 2 + epsilon * A ** 3

    X = defaultdict(list)
    G = graph
    Gedges = list(nx.non_edges(G))  # 获取图G中所有未连接的边并形成列表
    nodeList = list(G.nodes())
    nodeIndex = {node: idx for idx, node in enumerate(nodeList)}

    # G.degree() 用于计算图中每个节点的加权度数，加权度数是指一个节点所有边的权重之和
    Ki = dict(G.degree())
    Wi = dict(G.degree(weight='weight'))
    LPI = local_path(G, nodeList)
    L3 = l3_path(G, nodeList)
    WLPI = weighted_local_path(G, nodeList)

    for j, e in enumerate(Gedges):
        u, v = e
        common_ns = list(nx.common_neighbors(G, u, v))  # 用于找出G中两个节点共同的邻居节点

        # 计算方法不用管，这种通常是用来衡量两个节点在加权网络中的关系强度，选择较小值是为了保守估计u和v之间的连接强度
        w_common_ns = sum([min(G[u][z]['weight'], G[v][z]['weight']) for z in common_ns])

        # 将u和v各自的邻居节点返回，并求这两个的邻居节点并集
        union_ns = set(G.neighbors(u)) | set(G.neighbors(v))

        # 节点权重的并集
        w_union_ns = Wi[u] + Wi[v] - w_common_ns

        # if w_union_ns == 0:
        #     print(Wi[u], Wi[v], [min(G[u][z]['weight'], G[v][z]['weight']) for z in common_ns])
        X['Edge'].append(e)
        # X['Year'].append(year)

        X['Common Neighbor'].append(len(common_ns))  ####
        X['Weighted Common Neighbor'].append(w_common_ns)  ####

        # Ki[u]或Ki[v]等于0，是两者中可能存在孤立节点的情况
        # 如果不为零，我们依然关注的是没有连接的边，因为当前节点可能和任何一个其他邻居节点存在连接，那么这个节点的度还是有特征能用的

        # X['Weighted Leicht Holme Newman'].append(w_common_ns/(Wi[u]*Wi[v]))  ####
        if Ki[u] == 0 or Ki[v] == 0:  # 也就是这个节点是个孤立节点
            X['Salton'].append(0)
            X['Leicht Holme Newman'].append(0)
            X['Preferential Attachment'].append(0)
        else:
            X['Salton'].append(len(common_ns)/math.sqrt(Ki[u]*Ki[v]))  ####
            X['Leicht Holme Newman'].append(len(common_ns) / (Ki[u] * Ki[v]))  ####
            X['Preferential Attachment'].append(Ki[u] * Ki[v])  ####

        if Wi[u] == 0 or Wi[v] == 0:
            X['Weighted Leicht Holme Newman'].append(0)
            X['Weighted Salton'].append(0)
            X['Weighted Preferential Attachment'].append(0)
        else:
            X['Weighted Salton'].append(w_common_ns/math.sqrt(Wi[u] * Wi[v]))  ####
            X['Weighted Preferential Attachment'].append(Wi[u] * Wi[v])  ####
            X['Weighted Leicht Holme Newman'].append(w_common_ns / (Wi[u] * Wi[v]))

        if Ki[u] + Ki[v] == 0:
            X['Sorensen'].append(0)
        else:
            X['Sorensen'].append(2*len(common_ns)/(Ki[u]+Ki[v]))  ####

        if Wi[u] + Wi[v] == 0:
            X['Weighted Sorensen'].append(0)
        else:
            X['Weighted Sorensen'].append(2*w_common_ns/(Wi[u]+Wi[v]))  ####

        if min(Ki[u], Ki[v]) == 0:
            X['Hub Promoted'].append(0)
        else:
            X['Hub Promoted'].append(len(common_ns) / min(Ki[u], Ki[v]))

        if min(Wi[u], Wi[v]) == 0:
            X['Weighted Hub Promoted'].append(0)
        else:
            X['Weighted Hub Promoted'].append(w_common_ns/min(Wi[u], Wi[v]))  ####

        if max(Ki[u], Ki[v]) == 0:
            X['Hub Depressed'].append(0)
        else:
            X['Hub Depressed'].append(len(common_ns)/max(Ki[u], Ki[v]))  ####

        if max(Wi[u], Wi[v]) == 0:
            X['Weighted Hub Depressed'].append(0)
        else:
            X['Weighted Hub Depressed'].append(w_common_ns/max(Wi[u], Wi[v]))  ####

        X['Local Path'].append(LPI[nodeIndex[u], nodeIndex[v]])  ####
        X['L3 Path'].append(L3[nodeIndex[u], nodeIndex[v]])
        X['Weighted Local Path'].append(WLPI[nodeIndex[u], nodeIndex[v]])  ####
        if len(common_ns) > 0:
            X['Resource Allocation'].append(sum([1/Ki[z] for z in common_ns]))  ####
            # X['Weighted Resource Allocation'].append(w_common_ns*sum([1/Wi[z] for z in common_ns]))  ####
            X['Weighted Resource Allocation'].append(w_common_ns * sum([
                1 / Wi[z] if Wi[z] > 0 else 0
                for z in common_ns
            ])
                                                     )  ####

            X['Adamic Adar'].append(sum([1/math.log(Ki[z]) for z in common_ns]))  ####
            # X['Weighted Adamic Adar'].append(w_common_ns * sum([1 / math.log(Wi[z] + 1) for z in common_ns]))  ####
            X['Weighted Adamic Adar'].append(
                w_common_ns * sum([
                    1 / math.log(Wi[z] + 1) if math.log(Wi[z] + 1) > 0 else 0
                    for z in common_ns
                ])
            )
            # for z in common_ns:
            #     if math.log(Wi[z]+1) > 0:
            #         X['Weighted Adamic Adar'].append(w_common_ns*sum([1/math.log(Wi[z]+1)]))  ####
            #     else:
            #         X['Weighted Adamic Adar'].append(0)

            # X['Jaccard'].append(len(common_ns)/len(union_ns))  ####
            # X['Weighted Jaccard'].append(w_common_ns/w_union_ns)  ####
        else:
            X['Resource Allocation'].append(0)  ####  ####
            X['Weighted Resource Allocation'].append(0)
            X['Adamic Adar'].append(0)  ####
            X['Weighted Adamic Adar'].append(0)  ####
            # X['Jaccard'].append(0)  ####
            # X['Weighted Jaccard'].append(0)  ####

        if len(union_ns) > 0:
            X['Jaccard'].append(len(common_ns)/len(union_ns))  ####
        else:
            X['Jaccard'].append(0)  ####

        if w_union_ns > 0:
            X['Weighted Jaccard'].append(w_common_ns/w_union_ns)  ####
        else:
            X['Weighted Jaccard'].append(0)  ####

        # Removed edges
        # X['Removed'].append(e not in Hedges)

        # Added edges
        # X['Added'].append(e not in Hedges)
        # -1 -> 0: remove
        # 0 -> 1: none
        # 1 -> 2: add
        # X['Dynamic'].append(2 if e not in Hedges else 1)
        # # X['Gravity'].append(Gra[j])

        # X['Curr Weight'].append(G[u][v]['weight'])
        # X['Next Weight'].append(H[u][v]['weight'] if e in Hedges else 0)

        # X['Curr FWeight'].append(G[u][v]['weight'] / G.size(weight='weight'))
        # X['Next FWeight'].append(H[u][v]['weight'] / H.size(weight='weight') if e in Hedges else 0)

        # 因为在G中，当前边不存在，因此weight=0
        X['Curr Weight'].append(0)
        # X['Next Weight'].append(0 if e in Hedges else H[u][v]['weight'])
        # if is_edge_in_hedges(e, Hedges):
        #     # G存储的是当前时刻所有未连接的边，H存储的是下一时刻所有未连接的边
        #     # 当前时刻未连接，所以weight=0，下一时刻如果也未连接，则会出现在H中，因此weight也是0，如果下一时刻连接了，那么H中是找不到的
        #     X['Next Weight'].append(0)
        # else:
        #     X['Next Weight'].append(H[u][v]['weight'])

        X['Curr FWeight'].append(0 / G.size(weight='weight'))
        # X['Next FWeight'].append(0 if e in Hedges else H[u][v]['weight'] / H.size(weight='weight'))
        # if is_edge_in_hedges(e, Hedges):
        #     X['Next FWeight'].append(0)
        # else:
        #     X['Next FWeight'].append(H[u][v]['weight'] / H.size(weight='weight'))

    df = pd.DataFrame(X)
    return df


def get_feature_vector_remove(graph):   # 从给定的图 (Graph) 中提取多种特征向量
    def local_path(G, nodeList, epsilon = 0.01):
        A = nx.adjacency_matrix(G, nodelist=nodeList, weight = None).todense()
        return A**2+epsilon*A**3

    def weighted_local_path(G, nodeList, epsilon = 0.01):
        A = nx.adjacency_matrix(G, nodelist=nodeList, weight='weight').todense()
        return A**2+epsilon*A**3

    X = defaultdict(list)
    G = graph
    Gedges = list(G.edges())
    nodeList = list(G.nodes())       # 用于建立节点到索引的映射，从矩阵中索引特定节点对的路径数。
    nodeIndex = {node: idx for idx, node in enumerate(nodeList)}

    Ki = dict(G.degree())
    Wi = dict(G.degree(weight='weight'))
    LPI = local_path(G, nodeList)
    WLPI = weighted_local_path(G, nodeList)
    for j, e in enumerate(Gedges):
        u, v = e
        common_ns = list(nx.common_neighbors(G, u, v))
        w_common_ns = sum([min(G[u][z]['weight'], G[v][z]['weight']) for z in common_ns])
        union_ns = set(G.neighbors(u)) | set(G.neighbors(v))
        w_union_ns = Wi[u] + Wi[v] - w_common_ns
        if w_union_ns == 0:
            print(Wi[u], Wi[v], [min(G[u][z]['weight'], G[v][z]['weight']) for z in common_ns])
        X['Edge'].append(e)

        X['Common Neighbor'].append(len(common_ns))
        X['Weighted Common Neighbor'].append(w_common_ns)

        # Ki[u]或Ki[v]等于0，是两者中可能存在孤立节点的情况
        # 如果不为零，我们依然关注的是没有连接的边，因为当前节点可能和任何一个其他邻居节点存在连接，那么这个节点的度还是有特征能用的

        # X['Salton'].append(len(common_ns)/math.sqrt(Ki[u]*Ki[v]))
        # X['Weighted Salton'].append(w_common_ns/math.sqrt(Wi[u]*Wi[v]))
        #
        # X['Leicht Holme Newman'].append(len(common_ns) / (Ki[u] * Ki[v]))
        # X['Weighted Leicht Holme Newman'].append(w_common_ns / (Wi[u] * Wi[v]))
        #
        # X['Preferential Attachment'].append(Ki[u] * Ki[v])
        # X['Weighted Preferential Attachment'].append(Wi[u] * Wi[v])
        if Ki[u] == 0 or Ki[v] == 0:  # 也就是这个节点是个孤立节点
            X['Salton'].append(0)
            X['Leicht Holme Newman'].append(0)
            X['Preferential Attachment'].append(0)
        else:
            X['Salton'].append(len(common_ns) / math.sqrt(Ki[u] * Ki[v]))  ####
            X['Leicht Holme Newman'].append(len(common_ns) / (Ki[u] * Ki[v]))  ####
            X['Preferential Attachment'].append(Ki[u] * Ki[v])  ####

        if Wi[u] == 0 or Wi[v] == 0:
            X['Weighted Leicht Holme Newman'].append(0)
            X['Weighted Salton'].append(0)
            X['Weighted Preferential Attachment'].append(0)
        else:
            X['Weighted Salton'].append(w_common_ns / math.sqrt(Wi[u] * Wi[v]))  ####
            X['Weighted Preferential Attachment'].append(Wi[u] * Wi[v])  ####
            X['Weighted Leicht Holme Newman'].append(w_common_ns / (Wi[u] * Wi[v]))

        # X['Sorensen'].append(2*len(common_ns)/(Ki[u]+Ki[v]))
        # X['Weighted Sorensen'].append(2*w_common_ns/(Wi[u]+Wi[v]))

        if Ki[u] + Ki[v] == 0:
            X['Sorensen'].append(0)
        else:
            X['Sorensen'].append(2 * len(common_ns) / (Ki[u] + Ki[v]))  ####

        if Wi[u] + Wi[v] == 0:
            X['Weighted Sorensen'].append(0)
        else:
            X['Weighted Sorensen'].append(2 * w_common_ns / (Wi[u] + Wi[v]))  ####

        # X['Hub Promoted'].append(len(common_ns)/min(Ki[u],Ki[v]))
        # X['Weighted Hub Promoted'].append(w_common_ns/min(Wi[u],Wi[v]))

        if min(Ki[u], Ki[v]) == 0:
            X['Hub Promoted'].append(0)
        else:
            X['Hub Promoted'].append(len(common_ns) / min(Ki[u], Ki[v]))

        if min(Wi[u], Wi[v]) == 0:
            X['Weighted Hub Promoted'].append(0)
        else:
            X['Weighted Hub Promoted'].append(w_common_ns / min(Wi[u], Wi[v]))  ####

        # X['Hub Depressed'].append(len(common_ns)/max(Ki[u],Ki[v]))
        # X['Weighted Hub Depressed'].append(w_common_ns/max(Wi[u],Wi[v]))

        # if max(Ki[u], Ki[v]) == 0:
        #     X['Hub Depressed'].append(0)
        #     X['Weighted Hub Depressed'].append(0)
        # else:
        #     X['Hub Depressed'].append(len(common_ns) / max(Ki[u], Ki[v]))  ####
        #     X['Weighted Hub Depressed'].append(w_common_ns / max(Wi[u], Wi[v]))  ####

        if max(Ki[u], Ki[v]) == 0:
            X['Hub Depressed'].append(0)
        else:
            X['Hub Depressed'].append(len(common_ns) / max(Ki[u], Ki[v]))  ####

        if max(Wi[u], Wi[v]) == 0:
            X['Weighted Hub Depressed'].append(0)
        else:
            X['Weighted Hub Depressed'].append(w_common_ns / max(Wi[u], Wi[v]))  ####

        X['Local Path'].append(LPI[nodeIndex[u],nodeIndex[v]])
        X['Weighted Local Path'].append(WLPI[nodeIndex[u],nodeIndex[v]])
        if len(common_ns) > 0:
            X['Resource Allocation'].append(sum([1 / Ki[z] for z in common_ns]))  ####
            # X['Weighted Resource Allocation'].append(w_common_ns*sum([1/Wi[z] for z in common_ns]))  ####
            X['Weighted Resource Allocation'].append(w_common_ns * sum([
                1 / Wi[z] if Wi[z] > 0 else 0
                for z in common_ns
            ])
                                                     )  ####

            X['Adamic Adar'].append(sum([1 / math.log(Ki[z]) for z in common_ns]))  ####
            # X['Weighted Adamic Adar'].append(w_common_ns * sum([1 / math.log(Wi[z] + 1) for z in common_ns]))  ####
            X['Weighted Adamic Adar'].append(
                w_common_ns * sum([
                    1 / math.log(Wi[z] + 1) if math.log(Wi[z] + 1) > 0 else 0
                    for z in common_ns
                ])
            )
            # for z in common_ns:
            #     if math.log(Wi[z]+1) > 0:
            #         X['Weighted Adamic Adar'].append(w_common_ns*sum([1/math.log(Wi[z]+1)]))  ####
            #     else:
            #         X['Weighted Adamic Adar'].append(0)

            # X['Jaccard'].append(len(common_ns)/len(union_ns))  ####
            # X['Weighted Jaccard'].append(w_common_ns/w_union_ns)  ####
        else:
            X['Resource Allocation'].append(0)  ####  ####
            X['Weighted Resource Allocation'].append(0)
            X['Adamic Adar'].append(0)  ####
            X['Weighted Adamic Adar'].append(0)  ####
            # X['Jaccard'].append(0)  ####
            # X['Weighted Jaccard'].append(0)  ####

        if len(union_ns) > 0:
            X['Jaccard'].append(len(common_ns) / len(union_ns))  ####
        else:
            X['Jaccard'].append(0)  ####

        if w_union_ns > 0:
            X['Weighted Jaccard'].append(w_common_ns / w_union_ns)  ####
        else:
            X['Weighted Jaccard'].append(0)  ####

        X['Curr Weight'].append(G[u][v]['weight'])
        X['Curr FWeight'].append(G[u][v]['weight']/G.size(weight='weight'))
    df = pd.DataFrame(X)
    return df


def get_edge_slice(data, f_train_e=0.7, seed=30):    # 将数据集分割为训练集和测试集
    df = data
    edges = set(df.Edge.unique())
    random.seed(seed)
    # print(f"int(f_train_e*len(edges)) = {int(f_train_e*len(edges))}")
    edge_train = set(random.sample(list(edges), int(f_train_e*len(edges))))
    # edge_train = set(random.sample(list(edges), min(int(f_train_e * len(edges)), len(list(edges)))))
    # edge_train = set(random.sample(edges, int(f_train_e*len(edges))))
    edge_test = set([e for e in edges if e not in edge_train])
    df_se = df.loc[df['Edge'].isin(edge_train)].drop(columns=['Edge'])
    df_de = df.loc[df['Edge'].isin(edge_test)].drop(columns=['Edge'])
    return df_se, df_de


def df_to_XY(df, features, target='Dynamic'):   # 将数据框转换为机器学习模型的输入格式
    if 'Year' in df.columns:
        df = df.drop(columns = ['Year'])
    if "Edge" in df.columns:
        df = df.drop(columns = ['Edge'])
    X = df.loc[:, features].to_numpy()
    y = df.loc[:, df.columns == target].to_numpy()
    return X, y, df.loc[:, df.columns == 'Next Weight'].to_numpy()


'''
nx.difference(GI0, GI1).edges():
这表示取 GI0 相对于 GI1 的差集，即找出所有仅存在于 GI0 中的边。
这可以理解为从 GI0 到 GI1 的过渡中被删除的边。
其实就是找出真正remove的边是哪些

nx.difference(GI1, GI0).edges():
这表示取 GI1 相对于 GI0 的差集，即找出所有仅存在于 GI1 中的边。
这可以理解为从 GI0 到 GI1 的过渡中新增加的边。
其实是找出真正add的边是哪些
'''


def a_edges(graphs, inp_graph, time_idx):  # 识别在下一时刻存在，在当前时刻不存在的边，也就是真正add的边
    GI0, GI1 = graphs[time_idx+1], graphs[time_idx]
    GI0.add_nodes_from([n for n in GI1 if n not in GI0])
    GI1.add_nodes_from([n for n in GI0 if n not in GI1])
    added_edges = set(nx.difference(GI0, GI1).edges())  # 原来是list
    return added_edges


def r_edges(graphs, time_idx):  # 识别下一时刻不存在，在当前时刻存在的边
    GI0, GI1 = graphs[time_idx], graphs[time_idx+1]
    GI0.add_nodes_from([n for n in GI1 if n not in GI0])
    GI1.add_nodes_from([n for n in GI0 if n not in GI1])  # 这里两句是为了对齐两个图
    real_remove = set(nx.difference(GI0, GI1).edges())  # 这里是找出存在于GI0但不存在于GI1的边，也就是真正remove的边，这里原来写法是list
    return real_remove


####################################################

def main(year_start):  # 数据处理和网络分析的主流程

    if year_start.month == 1: return
    data = pd.read_csv('data/CN_Air_2014_2024_2.csv', sep=';')

    # data = pd.read_csv('data/test.csv', sep=';')
    data.set_index(['YEAR', 'MONTH'], inplace=True)

    data = data[data.source != data.target]
    nodes = set(data.source) & set(data.target)
    data = data[data.weight != 0]

    year = list(data.index.get_level_values(0).unique())
    month = list(data.index.get_level_values(1).unique())

    graphs_air = []
    air_dates = []
    for y in year:
        for m in month:
            if y == 2024 and m == 7:
                break
            df = data.loc[y, m]
            air_dates.append(date(y, m, 1))
            G = nx.from_pandas_edgelist(df, edge_attr=True)
            G.add_nodes_from(nodes)
            graphs_air.append(G)

    data = pd.read_csv('data/features/cnair_2014_2024_2.csv')  # 先试一下这个数据，没确定最终版

# ##########################

    out = {}
    idx = air_dates.index(year_start)
    train, test = get_edge_slice(data)
    features = ['Common Neighbor', 'Salton', 'Jaccard', 'Sorensen', 'Hub Promoted',
                'Hub Depressed', 'Leicht Holme Newman', 'Preferential Attachment',
                'Adamic Adar', 'Resource Allocation', 'Local Path']
    X_train, y_train, _ = df_to_XY(train[train.Year == str(year_start)], features)
    ros = RandomUnderSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)
    model_btf = XGBClassifier()
    model_btf.fit(X_train, y_train)
    model = model_btf
    diff_btf_remove = []
    diff_btf_add = []
    graphs_btf = [graphs_air[idx]]

    for i in tqdm(range(0, 36)):

        G = graphs_btf[i].copy()
        # print(
        #     f"idx = {idx}, G.number_of_edges() = {G.number_of_edges()}, graphs_air[idx+i+1].number_of_edges() = {graphs_air[idx + i + 1].number_of_edges()}")

        # 这部分是原始remove的代码的操作，目的是操作所有已连接的边，和feature_extractor_add.py文件对应
        df_remove = get_feature_vector_remove(G)
        edges_remove, X_remove = df_remove['Edge'].to_numpy(), df_remove[features].to_numpy()

        # 这部分是我们新增的add的代码的操作，目的是操作所有未连接的边，和feature_extractor_remove.py文件对应
        df_add = get_feature_vector_add(G)
        edges_add, X_add = df_add['Edge'].to_numpy(), df_add[features].to_numpy()

        real_removal = r_edges(graphs_air, idx+i)   # 找出真正remove的边
        real_added = a_edges(graphs_air, G, idx+i)     # 找出真正add的边

        pred_prob_remove22 = model.predict(X_remove).T[0]
        pred_prob_remove = model.predict_proba(X_remove).T[0]  # 模型预测出的remove的边
        pred_prob_add = model.predict_proba(X_add).T[2]     # 模型预测出的add的边

        # 通过循环遍历，逐步把下一时刻真正需要remove的边添加到G中
        N_add = len(real_added)

        # added 的操作
        added = zip(edges_add, pred_prob_add)
        added = sorted(added, key=lambda x: x[1])[0: N_add]
        added_edges = [i for i, _ in added]
        diff_btf_add.append(len(set(added_edges) & real_added) / N_add)
        # print('diff_btf_add: ', diff_btf_add)

        for u, v in added_edges:
            G.add_edge(u, v, weight=99)  # 我们假设weight不已知，这也是下一步做预测的前提

        N_remove = len(real_removal)

        removal = zip(edges_remove, pred_prob_remove)
        # [0: N_remove] 这部分是对排序后的列表进行切片操作，以选取前 N_remove 个元素
        removal = sorted(removal, key=lambda x: x[1])[0: N_remove]
        removed_edges = [i for i, _ in removal]
        diff_btf_remove.append(len(set(removed_edges) & real_removal) / N_remove)
        # print('diff_btf_remove: ', diff_btf_remove)

        G.remove_edges_from(removed_edges)  # G是下一时刻的图，G减去需要remove的边，则为下一时刻真正的图
        print(f"btf, i={i}, N_add={N_add}, N_remove={N_remove}, N_edges={len(list(G.edges()))}")
        graphs_btf.append(G.copy())

####################################################

    # best_params = {'lambda': 0.5650701862593042, 'alpha': 0.0016650896783581535,
    #        'colsample_bytree': 1.0, 'subsample': 0.5, 'learning_rate': 0.009,
    #        'n_estimators': 625, 'objective': 'reg:squarederror', 'max_depth': 5, 'min_child_weight': 6}
    best_params = {'lambda': 0.5650701862593042, 'alpha': 0.0016650896783581535,
                   'colsample_bytree': 1.0, 'subsample': 0.5, 'learning_rate': 0.009,
                   'n_estimators': 625, 'objective': 'multi:softmax', 'max_depth': 5, 'min_child_weight': 6,
                   'num_class': 3}  # 添加类别总数为3

    features = ['Curr Weight']
    X_train, y_train, y_reg = df_to_XY(train[train.Year == str(year_start)], features)
    ros = RandomUnderSampler()
    X_resample, y_resample = ros.fit_resample(X_train, y_train)

    model_www = XGBClassifier()
    model_www.fit(X_resample, y_resample)

    model_cls = XGBClassifier(**best_params)
    model_cls.fit(X_train, y_train)

    # model_reg = XGBRegressor(**best_params)
    # model_reg.fit(X_train, y_reg)
    model = model_www
    diff_www_add = []
    diff_www_remove = []
    graphs_www = [graphs_air[idx]]

    for i in tqdm(range(0, 36)):
        G = graphs_www[i].copy()
        for u, v in G.edges():
            G[u][v]['weight'] = model_cls.predict([G[u][v]['weight']])[0]

        df_remove = get_feature_vector_remove(G)
        edges_remove, X_remove = df_remove['Edge'].to_numpy(), df_remove[features].to_numpy()

        # added
        df_add = get_feature_vector_add(G)
        edges_add, X_add = df_add['Edge'].to_numpy(), df_add[features].to_numpy()

        real_removal = r_edges(graphs_air, idx + i)  # remove
        real_added = a_edges(graphs_air, G, idx + i)

        pred_prob_remove = model.predict_proba(X_remove).T[0]
        pred_prob_add = model.predict_proba(X_add).T[2]

        N_add = len(real_added)

        # added
        added = zip(edges_add, pred_prob_add)
        added = sorted(added, key=lambda x: x[1])[0: N_add]
        added_edges = [i for i, _ in added]
        diff_www_add.append(len(set(added_edges) & real_added) / N_add)

        for u, v in added_edges:
            G.add_edge(u, v, weight=99)  # 我们假设weight不已知，这也是下一步做预测的前提

        N_remove = len(real_removal)

        removal = zip(edges_remove, pred_prob_remove)
        removal = sorted(removal, key=lambda x: x[1])[0: N_remove]
        removed_edges = [i for i, _ in removal]

        diff_www_remove.append(len(set(removed_edges) & real_removal) / N_remove)

        G.remove_edges_from(removed_edges)
        print(f"www, i={i}, N_add={N_add}, N_remove={N_remove}, N_edges={len(list(G.edges()))}")
        graphs_www.append(G.copy())


        #####优化模型，代入前面预测的结果，处理、保存后面预测的输入###

        # """
        # 循环**个月：
        #         预测模型，输入是相似性指标；输出是新增/删减边、或者权重(乘客等)
        #         优化模型，输入是前面一个月预测的结果；输出是决策。用决策来计算后面一个月预测的相似性特征，再输入预测模型
        # """

#######################################
    btf_diff = []
    www_diff = []
    for i in range(0, 36):
        G = set(graphs_air[idx+i].edges())
        H = set(graphs_btf[i].edges())
        btf_diff.append(len(G & H)/len(G))
        H = set(graphs_www[i].edges())
        www_diff.append(len(G & H)/len(G))
    out[year_start] = (diff_btf_add, diff_btf_remove, diff_www_add, diff_www_remove, btf_diff, www_diff)
    import pickle
    with open(f'./results/' + f'{str(year_start)}pred_36' + '.pkl', 'wb') as f:
        pickle.dump(out, f)

# #########################

    out = {}
    idx = air_dates.index(year_start)
    diff_null_remove = []
    diff_null_add = []
    graphs_null = [graphs_air[idx]]

    for i in tqdm(range(0, 36)):
        # print('line 498: ', i)
        G = graphs_null[i].copy()

        # edges = list(G.edges())
        # 这部分是原始remove的代码的操作，目的是操作所有已连接的边，和feature_extractor_add.py文件对应
        df_remove = get_feature_vector_remove(G)
        edges_remove = df_remove['Edge'].to_numpy()

        # 这部分是我们新增的add的代码的操作，目的是操作所有未连接的边，和feature_extractor_remove.py文件对应
        df_add = get_feature_vector_add(G)
        edges_add = df_add['Edge'].to_numpy()

        real_removal = r_edges(graphs_air, idx + i)
        real_added = a_edges(graphs_air, G, idx + i)

        N_add = abs(len(real_added))

        # added_edges = a_edges(graphs_air, G, idx+i)

        # N_remove = G.number_of_edges() - graphs_air[idx+i+1].number_of_edges()
        # N_remove = abs(N_remove)

        N_remove = abs(len(real_removal))
        # print('line 512: ', i, N_remove)  # 16: 72

        if N_remove <= 0:
            # 处理负数样本大小的情况，例如将其设置为0或引发一个错误
            N_remove = 1

        if N_add <= 0:
            # 处理负数样本大小的情况，例如将其设置为0或引发一个错误
            N_add = 1

        if N_remove >= len(list(edges_remove)):
            N_remove = len(list(edges_remove))
        # print('line 524 edges: ', len(list(edges)))

        if N_add >= len(list(edges_add)):
            N_add = len(list(edges_add))

        # print('line 529: ', i, N_remove)  # 16: 0

        # print(f"N_remove = {N_remove}")

        # print(f"i={i}, N_add={N_add}, N_remove={N_remove}, N_edges={len(list(G.edges()))}")

        remove_edges = random.sample(list(edges_remove), N_remove)
        add_edges = random.sample(list(edges_add), N_add)

        # remove_edges = random.sample(edges, N_remove)
        # print('line 537: ', i, N_remove)
        diff_null_remove.append(len(set(remove_edges) & real_removal)/N_remove)
        diff_null_add.append(len(set(add_edges) & real_added)/N_add)

        for u, v in add_edges:
            G.add_edge(u, v, weight=99)

        G.remove_edges_from(remove_edges)

        print(f"null, i={i}, N_add={N_add}, N_remove={N_remove}, N_edges={len(list(G.edges()))}")

        graphs_null.append(G.copy())

    null_diff = []

    for i in range(0, 36):
        G = set(graphs_air[idx+i].edges())
        H = set(graphs_null[i].edges())
        btf_diff.append(len(G & H)/len(G))
    out[year_start] = (diff_null_remove, diff_null_add, null_diff)
    import pickle
    with open(f'./results/' + f'{str(year_start)}pred_36_null' +'.pkl', 'wb') as f:
        pickle.dump(out, f)


if __name__ == '__main__':
    data = pd.read_csv('data/CN_Air_2014_2024_2.csv', sep=';')
    data.set_index(['YEAR', 'MONTH'], inplace=True)
    data = data[data.source != data.target]
    nodes = set(data.source) & set(data.target)
    data = data[data.weight != 0]
    year = list(data.index.get_level_values(0).unique())
    month = list(data.index.get_level_values(1).unique())
    graphs_air = []
    air_dates = []
    for y in year:
        for m in month:
            if y == 2024 and m == 7:
                break
            df = data.loc[y, m]
            air_dates.append(date(y, m, 1))
            G = nx.from_pandas_edgelist(df, edge_attr=True)
            G.add_nodes_from(nodes)
            graphs_air.append(G)

    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = 10# multiprocessing.cpu_count()
    print(num_cores)
    results = Parallel(n_jobs=num_cores)(delayed(main)(year_start) for year_start in air_dates[:-36])
    bb = air_dates[:-1]
    # for year_start in air_dates[:-36]:
    #     print(year_start)
    #     main(year_start)



