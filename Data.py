import networkx as nx
from ToEmbedding import create_embedding  #,create_hot_encoding
import torch


# Funzione che legga il file .gml, costruisca il grafo, e restituisca: X (tensore per il training), Y (colonna outcomes), adj_matrix
def data(path):
    # Creazione del grafo leggendo il file .gml del database
    G = nx.read_gml(path, label='id')

    # Creazione grafo diretto
    DG = nx.DiGraph(G)

    # Matrice di adiacenza come array numpy
    adj_matrix = nx.adjacency_matrix(DG).todense()
    adj_matrix = torch.tensor(adj_matrix)

    # Estrazione dei links dei siti web dal database per ottenere gli embeddings del loro contenuto
    links = [DG.nodes.data()[id]['label'] for id in DG.nodes]

    links_embeddings = torch.tensor([])
    for l in links:
        links_embeddings = torch.cat((links_embeddings, create_embedding(l)), dim=0)

    # Estrazione e manipolazione dei blogs
    blogs_encoding = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])     # al posto di questo avremo il codice sottostante
    """
        # One-hot encoding dei blogs
        blogs = [DG.nodes.data()[id]['source'] for id in DG.nodes]
        blogs_encoding = [create_hot_encoding(b) for b in blogs]
    """

    # Estrazione della colonna degli outcomes
    values = [DG.nodes.data()[id]['value'] for id in DG.nodes]
    Y = torch.tensor(values)

    # Creazione tensore X (con i dati per il training)
    X = torch.cat((links_embeddings, blogs_encoding), dim=1)

    return X, Y, adj_matrix
