import networkx as nx
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from torch_geometric.utils import from_networkx


# Funzione che legga il file .gml, costruisca il grafo, e restituisca: X (tensore per il training), Y (colonna outcomes), edge_index
def load_data(path):
    # Creazione del grafo leggendo il file .gml del database
    G = nx.read_gml(path, label='id')

    # Creazione grafo diretto
    DG = nx.MultiDiGraph(G)

    # Estrazione dei links dei siti web dal database per ottenere gli embeddings del loro contenuto
    links = [DG.nodes.data()[id]['label'] for id in DG.nodes]

    # Creazione matrice degli input X
    links_embeddings = np.array([])
    X = np.array([np.append(links_embeddings, create_embedding(l)) for l in links])

    # Estrazione della colonna degli outcomes
    Y = np.array([DG.nodes.data()[id]['value'] for id in DG.nodes])

    # Matrice degli indici di adiacenza come tensore
    data = from_networkx(DG)
    edge_index = data.edge_index

    return X, Y, edge_index


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
