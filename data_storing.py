import networkx as nx
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.decomposition import PCA
import torch
import numpy as np
from torch_geometric.utils import from_networkx

# File to calculate the embeddings of the blog offline


# Function which create the embeddings starting from the textual content of the blogs
def create_embedding(url):

    url = "https://" + url          # https format for the url
    response = requests.get(url)    # HTTP Request to acquire the information from the corresponding website
    web_content = response.content  # Pull the resorse from the website

    # Extracting the text content from the HTML content using parsing
    soup = BeautifulSoup(web_content, "html.parser")
    texts = soup.stripped_strings       # delete formatting elements that are not relevant part of the text
    text_content = " ".join(texts)

    # BART model istanziation using the pipeline method of the transformers library to ensure summarization
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # chunks division to enable the summarization
    chunks_length = 3000        # optimal chunk dimension
    chunks = [text_content[i:i + chunks_length] for i in range(0, len(text_content), chunks_length)] # chunk division of the textual content

    # array which store in each cell the summarization of the chunks
    chunks_summaries = [chunk_error_handler(chunks[i], summarizer) for i in range(len(chunks))]

    # unifying the content of each chunks in each cells in one cell
    total_chunks_summary = " ".join(chunks_summaries)
    # managing the case of a small input length and a large input length to ensure consistancy in the summarization process
    if len(total_chunks_summary) > 1200 and len(total_chunks_summary) < chunks_length:
        summary = summarizer(total_chunks_summary, max_length=250, min_length=150, do_sample=False)[0]['summary_text']
    elif len(total_chunks_summary) > chunks_length:
        total_chunks_truncated = total_chunks_summary[:chunks_length]
        summary = summarizer(total_chunks_truncated, max_length=250, min_length=150, do_sample=False)[0]['summary_text']
    else:
        summary = total_chunks_summary

    # instantiate the model to calculate the embeddings 
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(summary, return_tensors='pt', truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    embedding = outputs.last_hidden_state.mean(dim=1)
    embedding = np.array(embedding)

    return embedding

# Function which identify the correct url which contains the resources
def suitable_url(url):
    try:
        response = requests.head(url, timeout=3)
        return response.status_code < 300           # filtering the urls which have a HTTP state code less than 300 to ensure the resources are acquired
    except requests.RequestException:
        return False

# function which summarize the chunks content checking the possible errors
def chunk_error_handler(chunk, summarizer):
    try:
        if len(chunk) > 1200:             # summarize the chunk only if have this number of characters
            chunk_summary = summarizer(chunk, max_length=250, min_length=150, do_sample=False)[0]['summary_text']
            return chunk_summary
        else:                             # returns the original chunk
            return chunk
    except IndexError:                # management of the chunks that have special characters which are not possible to summarize
        return " "                    # putting blank space instead of the chunks to preserve the string structure of the array

# Reading the graph dataset using the networkx library
Dataset = nx.read_gml('polblogs.gml', label='id')

# Graph creation using Multigraph type because of the duplicate edges in the dataset
MG = nx.MultiDiGraph(Dataset)

# Creating an empty Directed Graph in which are allowed directed edges, but not multiple ones
DG = nx.DiGraph()

# filling DG with nodes with suitable url
for node, data in MG.nodes(data=True):   # loading the nodes
    url = "https://" + data['label']
    if suitable_url(url):                # adding the link only if the is suitable
        DG.add_node(node, **data)

# filling DG with edges but taking into account cardinality (with weight parameter)
for source, target, data in MG.edges(data=True):  # loading the edges
    if DG.has_edge(source, target):          # increasing by one the weight of the edge if duplicated
        DG[source][target]['weight'] += 1
    elif DG.has_node(source) and DG.has_node(
            target):      # adding edges if and only if the source and the target nodes have been added previously ensuring consistency
        DG.add_edge(source, target, weight=1)


# creating and exporting the edge index matrix as a tensor
data = from_networkx(DG)
edge_index = data.edge_index
torch.save(edge_index, 'edge_index.pt')

# creating and exporting the edge weights matrix as a tensor
edge_weight = torch.tensor([DG[source][target]['weight'] for source, target in DG.edges], dtype=torch.float)
torch.save(edge_weight, 'edge_weight.pt')

# creating and exporting the value of the graph (Y)
Y = np.array([DG.nodes.data()[id]['value'] for id in DG.nodes])
np.save("Y.npy", Y)

# collecting the links to create the embeddings
links = [DG.nodes.data()[id]['label'] for id in DG.nodes]

links_embeddings = np.array([])
X = np.array([np.append(links_embeddings, create_embedding(l)) for l in links])

# reducing the embedding matrix using the PCA
pca = PCA(n_components=20)
X = pca.fit_transform(X)

# saving the embedding matrix
np.save("X.npy", X)

