import networkx as nx
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from torch_geometric.utils import from_networkx


# Funzione che legga il file .gml, costruisca il grafo, e restituisca: X (tensore per il training),
# Y (colonna outcomes), edge_index (adj_matrix compatta) e edge weight (cardinalità dei pesi)
def load_data(path):
    # Creazione del grafo leggendo il file .gml del database
    Dataset = nx.read_gml(path, label='id')

    # Creazione del grafo di tipo MultiDiGraph  per la presenza di archi duplicati
    MG = nx.MultiDiGraph(Dataset)

    # Trasformazione del MultiGraph in un semplice DiGraph, ma senza duplicati (con un peso per ogni edge)
    DG = nx.DiGraph()

    for node, data in MG.nodes(data=True):  # carico i nodi
        url = "https://" + data['label']
        if identify_broken_links(url):        # aggiungo il nodo solo se il link funziona
            DG.add_node(node, **data)

    for source, target, data in MG.edges(data=True):   # carico gli edges
        if DG.has_edge(source, target):     # aumento il peso dell'arco se già presente
            DG[source][target]['weight'] += 1
        elif DG.has_node(source) and DG.has_node(target):   # creo l'arco se non presente ma solo se i nodi source e target esistono nel DG (potrebbero non esistere poichè eliminiamo i link non funzionanti)
            DG.add_edge(source, target, weight=1)

    edge_weight = torch.tensor([DG[source][target]['weight'] for source, target in DG.edges], dtype=torch.float)

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

    return X, Y, edge_index, edge_weight


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

def identify_broken_links(url):
    if check_url(url):
        return True
    else:
        return False

def check_url(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code < 400
    except requests.RequestException:
        return False