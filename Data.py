import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from ToEmbedding import CreateEmbedding

# percorso del nostro database
file_path = './my_dataset.gml'

# Creazione del grafo leggendo il file .gml del database
G = nx.read_gml(file_path, label='id')

# Creazione grafo diretto
DG = nx.DiGraph(G)

# Estrazione dei links dei siti web dal database per ottenere gli embeddings del loro contenuto
links = [DG.nodes.data()[id]['label'] for id in DG.nodes]

to_embedd = CreateEmbedding()
embeddings = [to_embedd.Embedding(i) for i in links[:2]]
print(embeddings)

# Matrice di adiacenza come array numpy
adj_matrix = nx.adjacency_matrix(DG).todense()


