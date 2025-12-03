import math
import sys

import networkx as nx
import numpy as np
import pandas as pd

# 实现了特定目标的容器，以提供Python标准内建容器dict ,list , set , 和tuple的替代选择。defaultdict 是 dict 的子类
# 因此 defaultdict 也可被当成 dict 来使用，dict 支持的功能，defaultdict 基本都支持。
from collections import defaultdict
from datetime import date
from tqdm import tqdm  # 添加一个进度提示信息
from itertools import product  # 将itertools模块中的product()笛卡尔积函数导入了当前的命名空间中。


# sys.path.append('data')
# distant_df = pd.read_csv('data/us_air_distance.csv')
# distant_map = distant_df.set_index(['source_origin','target_origin']).to_dict()['distance']
# distant_map.update(distant_df.set_index(['target_origin','source_origin']).to_dict()['distance'])
# population_map = pd.read_csv('data/us_air_population_all.csv').set_index('Unnamed: 0').fillna(0).to_dict()['0']

# def get_gravitation(edges):
#     def my_divid(a,b):
#         if b==0 or a==0:
#             return None
#         else:
#             return a/b
#     Gra = []
#     for e in edges:
#         u, v = e
#         d = distant_map.get(e , 0)
#         ni = population_map.get(u, 0)
#         nj = population_map.get(v, 0)
#         Gra.append(my_divid(ni*nj, d**2))
#     meanv = np.mean([i for i in Gra if i])
#     return [i if i else meanv for i in Gra]


'''
对于大型图，邻接矩阵可能非常庞大，并且由于大量的0元素，会占用大量不必要的存储空间。
在这种情况下，保持其稀疏矩阵格式是一种内存效率更高的方式。
networkx.adjacency_matrix默认返回一个稀疏矩阵（来自scipy.sparse库），这对于处理大型图数据是有优势的。
使用.todense()或.toarray()可以将稀疏矩阵转换为NumPy数组，但这可能会导致大量的内存使用，特别是当图非常大时。
'''


def is_edge_in_hedges(edge, hedges):
    # Check both possible directions for the edge
    # 假设H是一个图的字典，H[u][v]['weight']是边(u, v)的权重
    # 假设Hedges是图H中所有边的集合
    # e是一个边的元组，如('fuo', 'kwl')
    # X['Next Weight']是一个列表，你需要向其中添加权重或0
    # (edge[1], edge[0]) in hedges: 这部分首先将 edge 元组的元素反转，即如果 edge 是 (u, v)，则反转后变为 (v, u)。
    # 之后，检查这个反转后的元组是否存在于 hedges 中。
    return edge in hedges or (edge[1], edge[0]) in hedges


def features_extractor_add(graphs, dates):
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
    for i in tqdm(range(len(graphs)-1)):
        # print(i)
        G, H = graphs[i], graphs[i+1]
        G.add_nodes_from([n for n in H if n not in G])
        H.add_nodes_from([n for n in G if n not in H])
        # Hedges = set(H.edges())  # 获取图H中所有已连接的边
        Hedges = set(nx.non_edges(H))  # 获取图H中所有未连接的边
        # Hedges = set(nx.non_edges(H))  # 获取图H中所有未连接的边
        # Gedges = list(G.edges())  # 获取图G中所有已连接的边并形成列表
        Gedges = list(nx.non_edges(G))  # 获取图G中所有未连接的边并形成列表
        # Gedges_set = set(G.edges())
        nodeList = list(G.nodes())
        nodeIndex = {node: idx for idx, node in enumerate(nodeList)}
        year = dates[i]

        # G.degree() 用于计算图中每个节点的加权度数，加权度数是指一个节点所有边的权重之和
        Ki = dict(G.degree())
        Wi = dict(G.degree(weight='weight'))
        LPI = local_path(G, nodeList)
        L3 = l3_path(G, nodeList)
        WLPI = weighted_local_path(G, nodeList)
        # Gra = get_gravitation(Gedges)

        # 用于计算两个图的差集: G = graphs[i]; H = graphs[i+1]
        added_edges = list(nx.difference(H, G).edges())
        # removed_edges = Hedges - Gedges_set

        for j, e in enumerate(Gedges):
            u, v = e
            common_ns = list(nx.common_neighbors(G, u, v))  # 用于找出G中两个节点共同的邻居节点

            # 计算方法不用管，这种通常是用来衡量两个节点在加权网络中的关系强度，选择较小值是为了保守估计u和v之间的连接强度
            w_common_ns = sum([min(G[u][z]['weight'], G[v][z]['weight']) for z in common_ns])

            # 将u和v各自的邻居节点返回，并求这两个的邻居节点并集
            union_ns = set(G.neighbors(u)) | set(G.neighbors(v))

            # 节点权重的并集
            w_union_ns = Wi[u] + Wi[v] - w_common_ns

            if w_union_ns == 0:
                print(Wi[u], Wi[v], [min(G[u][z]['weight'], G[v][z]['weight']) for z in common_ns])
            X['Edge'].append(e)
            X['Year'].append(year)

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
                X['Hub Depressed'].append(len(common_ns) / max(Ki[u], Ki[v]))  ####

            if max(Wi[u], Wi[v]) == 0:
                X['Weighted Hub Depressed'].append(0)
            else:
                X['Weighted Hub Depressed'].append(w_common_ns / max(Wi[u], Wi[v]))  ####

            X['Local Path'].append(LPI[nodeIndex[u], nodeIndex[v]])  ####
            X['L3 Path'].append(L3[nodeIndex[u], nodeIndex[v]])
            X['Weighted Local Path'].append(WLPI[nodeIndex[u], nodeIndex[v]])  ####
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

            # Removed edges
            # X['Removed'].append(e not in Hedges)

            # Added edges
            # X['Added'].append(e not in Hedges)
            # -1 -> 0: remove
            # 0 -> 1: none
            # 1 -> 2: add
            X['Dynamic'].append(2 if e not in Hedges else 1)
            # # X['Gravity'].append(Gra[j])

            # X['Curr Weight'].append(G[u][v]['weight'])
            # X['Next Weight'].append(H[u][v]['weight'] if e in Hedges else 0)

            # X['Curr FWeight'].append(G[u][v]['weight'] / G.size(weight='weight'))
            # X['Next FWeight'].append(H[u][v]['weight'] / H.size(weight='weight') if e in Hedges else 0)

            # 因为在G中，当前边不存在，因此weight=0
            X['Curr Weight'].append(0)
            # X['Next Weight'].append(0 if e in Hedges else H[u][v]['weight'])
            if is_edge_in_hedges(e, Hedges):
                # G存储的是当前时刻所有未连接的边，H存储的是下一时刻所有未连接的边
                # 当前时刻未连接，所以weight=0，下一时刻如果也未连接，则会出现在H中，因此weight也是0，如果下一时刻连接了，那么H中是找不到的
                X['Next Weight'].append(0)
            else:
                X['Next Weight'].append(H[u][v]['weight'])

            X['Curr FWeight'].append(0 / G.size(weight='weight'))
            # X['Next FWeight'].append(0 if e in Hedges else H[u][v]['weight'] / H.size(weight='weight'))
            if is_edge_in_hedges(e, Hedges):
                X['Next FWeight'].append(0)
            else:
                X['Next FWeight'].append(H[u][v]['weight'] / H.size(weight='weight'))

    df = pd.DataFrame(X)
    return df
