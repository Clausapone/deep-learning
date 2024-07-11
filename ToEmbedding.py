import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch


def create_embedding(url):

    # Scaricare il contenuto del sito web usando l'url fronito
    url = "https://" + url
    response = requests.get(url)    # richiesta HTTP tramite
    web_content = response.content

    # Estrarre il testo rilevante dal contenuto HTML
    soup = BeautifulSoup(web_content, "html.parser")
    texts = soup.stripped_strings
    text_content = " ".join(texts)

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

    return embeddings


"""
def create_hot_encoding(blogs):

    return encoded_blogs
"""
