import networkx as nx
from node2vec import Node2Vec
import os
import json
import pickle
import pandas as pd
with open(os.path.join('../Dataset','friends_adjacency_with_edge_weight.json'),'r') as f:

    friends_adjaency_matrix = json.load(f)

G=nx.Graph()


data = pd.read_csv(os.path.join('../Dataset','train_data.csv'))
users=data['user'].unique()[:1000]
print(users)
for keys in users:
    print(keys)
    for friend in friends_adjaency_matrix[str(keys)][:10]:
        G.add_edge(str(keys),str(friend[0]),weight=friend[1])

print("Graph Intialized")
EmbeddingNode = Node2Vec(G, dimensions=64, walk_length=20, num_walks=100, workers=1)

print("Training Started")
model = EmbeddingNode.fit(window=10, min_count=1, batch_words=4)
print("Training Completed")

with open("node2vec_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load back
with open("node2vec_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Example: Get vector for a node
print(model.wv['1'])  # node ids are strings

