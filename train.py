from utils import load_data
#from torch import optim
import torch.nn.functional as F
from torch.nn import BCELoss
from model import GCN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Uso una funzione per ottenere le strutture dati per il training, dando in input il percorso del nostro database
file_path = 'polblogs.gml'
X, Y, adj_matrix = load_data(file_path)

model = GCN(X.shape[1], 1)

model.train()
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = BCELoss()
    loss.backward()
    optimizer.step()
    return loss


