import networkx as nx
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.decomposition import PCA
import torch
import numpy as np
from torch_geometric.utils import from_networkx


"""
Modulo che legge il file .gml, costruisce il grafo, e restituisce: X (tensore per il training),
Y (colonna outcomes), edge_index (adj_matrix compatta) e edge weight (cardinalità dei pesi)
"""

# UTILS
#-----------------------------------------------------------------------------------------------------

# Funzione che identifichi gli url idonei in base allo status code e ad un timeout
def suitable_url(url):
    try:
        response = requests.head(url, timeout=3)
        return response.status_code < 300
    except requests.RequestException:
        return False

#-----------------------------------------------------------------------------------------------------
# Funzione che crei l'embedding del contenuto di un sito web, dato l'url
def create_embedding(url):
    # Scaricare il contenuto del sito web usando l'url fronito
    url = "https://" + url
    response = requests.get(url)  # richiesta HTTP per ottenere informazioni sul sito web
    web_content = response.content  # ricavare il contenuto del sito web

    # Estrarre il testo rilevante dal contenuto HTML tramite parsing
    soup = BeautifulSoup(web_content, "html.parser")
    texts = soup.stripped_strings  # eliminare elementi di formattazione che non fanno parte del testo
    text_content = " ".join(texts)  # introdurre gli spazi tra le parole

    # WEB CONTENT --> RIASSUNTO

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # divisione in chunks
    chunks_length = 3500
    chunks = [text_content[i:i + chunks_length] for i in range(0, len(text_content), chunks_length)]

    # riassunti dei chunks
    chunks_summaries = [summarizer(chunks[i], max_length=250, min_length=150, do_sample=False)[0]['summary_text'] for i in range(len(chunks))]

    # riassunto totale dei chunks
    total_chunks_summary = " ".join(chunks_summaries)
    if len(total_chunks_summary) < chunks_length:
        summary = summarizer(total_chunks_summary, max_length=250, min_length=150, do_sample=False)[0]['summary_text']
    else:
        total_chunks_truncated = total_chunks_summary[:chunks_length]
        summary = summarizer(total_chunks_truncated,  max_length=250, min_length=150, do_sample=False)[0]['summary_text']

    # RIASSUNTO --> LARGE EMBEDDING
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(summary, return_tensors='pt', truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    embedding = outputs.last_hidden_state.mean(dim=1)
    embedding = np.array(embedding)

    return embedding


# BODY
#-----------------------------------------------------------------------------------------------------

# Creazione del grafo leggendo il file .gml del database
Dataset = nx.read_gml('toy_dataset.gml', label='id')

# Creazione del grafo di tipo MultiDiGraph  per la presenza di archi duplicati
MG = nx.MultiDiGraph(Dataset)

# Trasformazione del MultiGraph in un semplice DiGraph, ma senza duplicati (con un peso per ogni edge)
DG = nx.DiGraph()

for node, data in MG.nodes(data=True):  # carico i nodi
    url = "https://" + data['label']
    if suitable_url(url):  # aggiungo il nodo solo se il link funziona
        DG.add_node(node, **data)

for source, target, data in MG.edges(data=True):  # carico gli edges
    if DG.has_edge(source, target):  # aumento il peso dell'arco se già presente
        DG[source][target]['weight'] += 1
    elif DG.has_node(source) and DG.has_node(
            target):  # creo l'arco se non presente ma solo se i nodi source e target esistono nel DG (potrebbero non esistere poichè eliminiamo i link non funzionanti)
        DG.add_edge(source, target, weight=1)


# CREAZIONE ed ESPORTAZIONE della matrice degli indici di adiacenza (tensore)
data = from_networkx(DG)
edge_index = data.edge_index
torch.save(edge_index, 'edge_index.pt')

# CREAZIONE ed ESPORTAZIONE della matrice dei pesi associati ad edge_index (tensore)
edge_weight = torch.tensor([DG[source][target]['weight'] for source, target in DG.edges], dtype=torch.float)
torch.save(edge_weight, 'edge_weight.pt')

# CREAZIONE ed ESPORTAZIONE  della colonna degli outcomes Y
Y = np.array([DG.nodes.data()[id]['value'] for id in DG.nodes])
np.save("Y.npy", Y)

# CREAZIONE ed ESPORTAZIONE della matrice degli inputs X ridotta tramite PCA (LARGE EMBEDDING --> FINAL SHORT EMBEDDING)

# Estrazione dei links dei siti web dal database per ottenere gli embeddings del loro contenuto
links = [DG.nodes.data()[id]['label'] for id in DG.nodes]

links_embeddings = np.array([])
X = np.array([np.append(links_embeddings, create_embedding(l)) for l in links])
np.save("large_embeddings.npy", X)

pca = PCA(n_components=6)   # AGGIORNARE
X = pca.fit_transform(X)
np.save("X.npy", X)
