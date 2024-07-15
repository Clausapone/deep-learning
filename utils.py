import networkx as nx
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


# Funzione che legga il file .gml, costruisca il grafo, e restituisca: X (tensore per il training), Y (colonna outcomes), adj_matrix
def load_data(path):
    # Creazione del grafo leggendo il file .gml del database
    G = nx.read_gml(path, label='id')

    # Creazione grafo diretto
    DG = nx.MultiDiGraph(G)

    # Matrice di adiacenza come array numpy
    adj_matrix = nx.adjacency_matrix(DG).todense()

    # Estrazione dei links dei siti web dal database per ottenere gli embeddings del loro contenuto
    links = [DG.nodes.data()[id]['label'] for id in DG.nodes]

    links_embeddings = np.array([])
    links_embeddings = np.array([np.append(links_embeddings, create_embedding(l)) for l in links])

    # Estrazione e manipolazione dei blogs
    blogs_encoding = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0],  [1, 1, 0],  [0, 1, 1],  [0, 0, 1],  [0, 1, 0]])    # al posto di questo avremo il codice sottostante
    """
        # One-hot encoding dei blogs
        blogs = [DG.nodes.data()[id]['source'] for id in DG.nodes]
        blogs_encoding = [create_hot_encoding(b) for b in blogs]
    """

    # Creazione tensore X (dati per il training)
    X = np.hstack((links_embeddings, blogs_encoding))

    # Estrazione della colonna degli outcomes
    Y = np.array([DG.nodes.data()[id]['value'] for id in DG.nodes])

    return X, Y, adj_matrix


# Funzione che crei l'embedding del contenuto di un sito web, dato l'url
def create_embedding(url):

    # Scaricare il contenuto del sito web usando l'url fronito
    url = "https://" + url
    response = requests.get(url)    # richiesta HTTP per ottenere informazioni sul sito web
    web_content = response.content      # ricavare il contenuto del sito web

    # Estrarre il testo rilevante dal contenuto HTML tramite parsing
    soup = BeautifulSoup(web_content, "html.parser")
    texts = soup.stripped_strings   # eliminare elementi di formattazione che non fanno parte del testo
    text_content = " ".join(texts)  # introdurre gli spazi tra le parole

    # Istanziare il modello di embedding di Hugging Face
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenizzare il testo
    inputs = tokenizer(text_content, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Ottenere l'embedding esteso
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # Rappresentazione compatta dell'embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings = np.array(embeddings)

    return embeddings


"""
# Funzione che codifichi un blog con one-hot encoding
def create_hot_encoding(blogs):


    return encoded_blogs
"""
