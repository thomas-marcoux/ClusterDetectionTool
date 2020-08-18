import community as community_louvain
import itertools
from networkx.algorithms.community.centrality import girvan_newman
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import os
from scipy.spatial import distance
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score

def remove_isolates(G):
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    return G

def group_by(df, col, id):
    res = {}
    groups = set(df[col])
    for g in groups:
        res[g] = list(df.loc[df[col] == g][id])
    return res

def get_similarity_index(list1, list2, percent=False):
    r = len(set(list1) & set(list2)) / float(len(set(list1) | set(list2))) * (100 if percent else 1)
    return r

def get_normal_mutual_info(x, y):
    return normalized_mutual_info_score(x, y)

def get_mutual_info(x, y):
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
    for i in range(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy)== 0:
            continue
        pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
        t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi

class Community():

    comentions_file = 'MPxMP_CoMentions.csv'
    topics_file = 'MPxTopic_Matrix.csv'
    mps_file = 'mps_%s.csv'
    layout = 'circular'

    def __init__(self, dir, convocation_id, community_alg, remove_singletons=False):
        self.dir = dir
        self.filename = self.comentions_file
        self.convocation_id = convocation_id
        self.community_alg = community_alg
        self.load_graph()
        self.load_mps()
        self.load_communities(remove_singletons)

    def load_graph(self):
        path = os.path.join(self.dir, self.filename)
        df = pd.read_csv(path, index_col=0)
        G = nx.from_pandas_adjacency(df)
        G = remove_isolates(G)
        self.G = G

    def load_mps(self):
        path = os.path.join(self.dir, self.mps_file % self.convocation_id)
        id = 'name_eng'
        group_columns = ['party', 'committee']
        df = pd.read_csv(path, usecols=[id] + group_columns)
        self.real_groups = {}
        for g in group_columns:
            self.real_groups[g] = group_by(df, g, id)

    def load_communities(self, remove_singletons=False):
        if self.community_alg == 'Girvan-Newman':
            nodes_iterator = girvan_newman(self.G)
            groups = tuple(sorted(c) for c in next(nodes_iterator))
            self.nodes = []
            [self.nodes.extend(l) for l in groups]
            self.communities = [i for i in range(0, len(groups))]
            self.detected_groups = {key: community for key, community in zip(self.communities, groups)}
        
        if self.community_alg == 'Louvain':
            partition = community_louvain.best_partition(self.G)
            self.communities = list(partition.values())
            self.nodes = partition.keys()
            members = list(self.nodes)
            self.detected_groups = {key : [members[idx] for idx in range(len(members)) if self.communities[idx]== key] for key in set(self.communities)}

        if remove_singletons:
            self.detected_groups = {key : members for key, members in self.detected_groups.items() if len(members) > 1}

    def write_detected(self):
        data = {key: ', '.join(MPs) for key, MPs in self.detected_groups.items()}
        df = pd.DataFrame.from_dict(data, orient='index')
        df.to_csv(os.path.join(self.dir, f'{self.community_alg}_Detected_Communities_{self.convocation_id}.csv'))

    def compute_similarity(self):
        data = []
        for group_type, real_groups in self.real_groups.items():
            for real_group_id, real_group in real_groups.items():
                for detected_group_id, detected_group in self.detected_groups.items():
                    jaccard = get_similarity_index(detected_group, real_group)
                    # mutual_info = get_mutual_info(detected_group, real_group)
                    # normal_sk_mutual_info = get_normal_mutual_info(detected_group, real_group)
                    data.append([detected_group_id, group_type, real_group_id, jaccard])
        df = pd.DataFrame(data, columns=['detected_group', 'real_group_type', 'real_group', 'jaccard'])
        df.sort_values('jaccard', inplace=True, ascending=False)
        df.to_csv(os.path.join(self.dir, f'{self.community_alg}_Similarity_{self.convocation_id}.csv'))

    def draw_figure(self):
        if self.community_alg != 'Louvain':
            raise(Exception('Community algortihm does not provide best partition'))
        G = self.G
        # Draw the graph
        pos = nx.circular_layout(G) if self.layout == 'circular' else nx.spring_layout(G)
        # Color the nodes according to their partition
        cmap = cm.get_cmap('viridis', max(self.communities) + 1)
        nx.draw_networkx_nodes(G, pos, self.nodes, node_size=40, cmap=cmap, node_color=self.communities)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        fig = plt.gcf()
        fig.savefig(os.path.join(self.dir, f'{self.community_alg}_{self.convocation_id}.png'))
        fig.clf()

    def output_degree(self):
        degrees = list(self.G.degree(self.nodes))
        df = pd.DataFrame(degrees)
        df = df.rename(columns={0: "MP", 1: "Degree"})
        if self.community_alg == 'Louvain':
            df['Community'] = self.communities
        df.sort_values("Degree", inplace=True, ascending=False)
        df.to_csv(os.path.join(self.dir, f'{self.community_alg}_Degrees_{self.convocation_id}.csv'))

def main():
    convocation_ids = [5,6,7,8]
    for convocation_id in convocation_ids:
        print(f"Convocation {convocation_id}...")
        dir = os.path.join(os.getcwd(), 'Data', 'Output_Convocation' + str(convocation_id))
        community = Community(dir, convocation_id, community_alg='Louvain', remove_singletons=True)
        # community = Community(dir, convocation_id, community_alg='Girvan-Newman', remove_singletons=True)
        community.write_detected()
        community.compute_similarity()
        community.output_degree()
        # community.draw_figure()

main()