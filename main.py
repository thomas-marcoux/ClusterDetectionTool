from collections import Counter
import community as community_louvain
import itertools
from networkx.algorithms.community.centrality import girvan_newman
from networkx.drawing.nx_agraph import write_dot
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from plotly.offline import plot
import random
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

def load_mp_info(dir, convocation_id, column, mp_id='name_eng'):
    mps_file = 'mps_%s.csv'
    path = os.path.join(dir, mps_file % convocation_id)
    df = pd.read_csv(path, usecols=[mp_id, column])
    df.rename(columns={column: f'convocation_{convocation_id}'}, inplace=True)
    df.set_index(mp_id, inplace=True)
    return df

def get_column(df, col, drop_na=False):
    return df[col].dropna(axis=0, how='any') if drop_na else df[col]

def draw_party_evolution(df, height=850, width=1540):
    drop_na = False
    links_df = pd.DataFrame(columns=['Source', 'Target', 'Value'])
    nodes_df = pd.DataFrame(columns=['Color', 'Node, Label'])
    r = lambda: random.randint(0,255)
    classes = {}
    colors = {}
    node_id = 0
    for i in range(0, df.columns.size):
        parties = list(df.iloc[:,i].unique())
        if np.nan not in parties:
            parties.append(np.nan)
        classes[i] = dict()
        for j in range(0, len(parties)):
            classes[i][parties[j]] = node_id
            if parties[j] not in colors:
                colors[parties[j]] = '#%02X%02X%02X' % (r(),r(),r())
            nodes_df.loc[nodes_df.shape[0]] = [colors[parties[j]], parties[j]]
            node_id += 1
    for i in range(0, df.columns.size - 1):
        for j in range(0, df.shape[0]):
            source = classes[i][df.iloc[j, i]]
            target = classes[i+1][df.iloc[j, i+1]]
            links_df.loc[links_df.shape[0]] = [source, target, 1]

    data_trace = dict(
        type='sankey',
        domain = dict(
            x =  [0,1],
            y =  [0,1]
        ),
        orientation = "h",
        valueformat = ".0f",
        node = dict(
            pad = 10,
            thickness = 30,
            line = dict(
                color = "black",
                width = 0.5
            ),
            label = get_column(nodes_df, 'Node, Label', drop_na),
            color = nodes_df['Color']
        ),
        link = dict(
            source = get_column(links_df, 'Source', drop_na),
            target = get_column(links_df, 'Target', drop_na),
            value = get_column(links_df, 'Value', drop_na),
        )
    )

    layout =  dict(
        title = "Ukrainian MPs party distribution from Convocation 5 to 8",
        height = height,
        width = width,
        font = dict(
            size = 10
        ),    
    )

    fig = dict(data=[data_trace], layout=layout)
    plot(fig, validate=False)

class Community():

    comentions_file = 'MPxMP_CoMentions.csv'
    topics_file = 'MPxTopic_Matrix.csv'
    mps_file = 'mps_%s.csv'
    layout = 'circular'

    def __init__(self, dir, convocation_id, source, community_alg, remove_singletons=False):
        self.dir = dir
        self.filename = self.comentions_file if source is 'comentions' else self.topics_file
        self.convocation_id = convocation_id
        self.source = source
        self.community_alg = community_alg
        self.load_graph()
        self.load_mps()
        self.load_communities(remove_singletons)

    def load_graph(self, keep_isolates=False, output=False):
        path = os.path.join(self.dir, self.filename)
        df = pd.read_csv(path, index_col=0)
        if self.source is 'comentions':
            G = nx.from_pandas_adjacency(df)
        elif self.source  is 'topics':
            G = nx.Graph()
            MPs = df.index.values.tolist()
            topics = list(df.columns)
            G.add_nodes_from(MPs, bipartite=0)
            G.add_nodes_from(topics, bipartite=1)
            weighted_edges = [(idx, topic, df.loc[idx, topic]) for topic in topics for idx, row in df.iterrows() if df.loc[idx, topic] > 0]
            G.add_weighted_edges_from(weighted_edges)
        if keep_isolates is False:
            G = remove_isolates(G)
        if output:
            nx.write_graphml_lxml(G, os.path.join(self.dir, f"{self.source}_{self.convocation_id}_graph.graphml"))
        self.G = G

    def load_mps(self):
        path = os.path.join(self.dir, self.mps_file % self.convocation_id)
        id = 'name_eng'
        group_columns = ['party', 'committee']
        df = pd.read_csv(path, usecols=[id] + group_columns)
        self.real_groups = {}
        for g in group_columns:
            self.real_groups[g] = group_by(df, g, id)
        df.set_index(id)
        self.mps = df.to_dict('index')
        self.mps = { dict[id]: dict for key, dict in self.mps.items()}

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
        data = {key: [', '.join(MPs), self.getAttrStr(MPs, 'party'), self.getAttrStr(MPs, 'committee')] for key, MPs in self.detected_groups.items()}
        df = pd.DataFrame.from_dict(data, orient='index')
        self.write_excel('Detected_Communities', df)

    def getAttrStr(self, ids, attr):
        data = self.getAttrList(ids, attr)
        return ', '.join(data) if len(data) > 0 else None

    def getAttrList(self, ids, attr):
        return [self.mps[id][attr] for id in ids if self.mps[id][attr] is not np.nan]

    def writeGroupCompositions(self):
        attrs = ['party', 'committee']
        subdir = 'composition'
        for key, MPs in self.detected_groups.items():
            for attr in attrs:
                data = self.getAttrList(MPs, attr)
                counts = Counter(data)
                plt.pie([float(v) for v in counts.values()], labels=[k for k in counts], autopct=None)
                # fig1, ax1 = plt.subplots()
                # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
                # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                fig = plt.gcf()
                fig.savefig(os.path.join(self.dir, subdir, f'{self.source}_{self.community_alg}_{self.convocation_id}_group{key}_{attr}_distribution.png'))
                fig.clf()

    def write_similarity(self):
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
        self.write_excel('Similarity', df)

    def draw_figure(self):
        if self.community_alg != 'Louvain':
            return
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
        fig.savefig(os.path.join(self.dir, f'{self.source}_{self.community_alg}_{self.convocation_id}.png'))
        fig.clf()

    def write_degree(self):
        degrees = list(self.G.degree(self.nodes))
        df = pd.DataFrame(degrees)
        df = df.rename(columns={0: "MP", 1: "Degree"})
        df['Community'] = self.communities
        df.sort_values("Degree", inplace=True, ascending=False)
        self.write_excel('Degrees', df)

    def write_excel(self, filename, df):
        with pd.ExcelWriter(os.path.join(self.dir, f'{filename}[{self.convocation_id}].xlsx'), engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name=f'{self.source}_{self.community_alg}')

def main():
    convocation_ids = [5,6,7,8]
    df_parties = pd.DataFrame()
    # community_algs = ['Girvan-Newman', 'Louvain']
    community_algs = ['Louvain']
    sources = ['comentions', 'topics']
    # sources = ['comentions']
    # sources = ['topics']
    for convocation_id in convocation_ids:
        print(f"Convocation {convocation_id}...")
        dir = os.path.join(os.getcwd(), 'Data', 'Output_Convocation' + str(convocation_id))
        df = load_mp_info(dir, convocation_id, 'party')
        df_parties = df_parties.join(df, how='outer') if not df_parties.empty else df
        # for source in sources:
        #     for community_alg in community_algs:
        #         print(f'{source.capitalize()} : {community_alg.capitalize()}')
        #         community = Community(dir, convocation_id, source, community_alg, remove_singletons=True)
                # community.write_detected()
                # community.writeGroupCompositions()
                # community.write_degree()
                # community.draw_figure()
                # community.write_similarity()
    draw_party_evolution(df_parties)

main()
