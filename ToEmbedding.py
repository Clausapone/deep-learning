import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch

class CreateEmbedding:

    def Embedding(self, url):

        # Step 1: Scaricare il contenuto del sito web
        url = "40ozblog.blogspot.com/"
        url = "https://" + url
        response = requests.get(url)
        web_content = response.content

        # Step 2: Estrarre il testo rilevante dal contenuto HTML
        soup = BeautifulSoup(web_content, "html.parser")
        texts = soup.stripped_strings
        text_content = " ".join(texts)

        # Step 3: Utilizzare un modello di embedding di Hugging Face
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Tokenizzare il testo
        inputs = tokenizer(text_content, return_tensors='pt', truncation=True, padding=True, max_length=512)

        # Ottenere l'embedding
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

        # Mediare gli embedding per ottenere una rappresentazione fissa del testo
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings

