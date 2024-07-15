import torch

from utils import load_data
from torch import optim
from torch.nn import BCELoss
from model import GCN
from train import train
from test import test
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import numpy as np

# Funzione per ottenere le strutture dati per il training, dando in input il percorso del nostro database
file_path = 'toy_dataset.gml'
X, Y, adj_matrix = load_data(file_path)

# Train and test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalizzazione
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creazione modello, loss e optimizer
model = GCN(X_train.shape[1])
criterion = BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)     # (VALUTARE: SGD, RMSProp, Adagrad,...)

# Training (prova senza CV)
X_train = torch.tensor(X_train)
Y_train = torch.tensor(Y_train)

train(model, X_train, Y_train, adj_matrix, optimizer, criterion, 1000)



"""
# Fase di Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#for combination in parametrers_combinations
scores = np.array([])
for fold, (train_mask, val_mask) in enumerate(kf.split(X_train)):
    X_train, X_val = X_train[train_mask], X_train[val_mask]
    Y_train, Y_val = Y_train[train_mask], Y_train[val_mask]

    train_adj_matrix = adj_matrix[np.ix_(train_mask, train_mask)]
    val_adj_matrix = adj_matrix[np.ix_(val_mask, val_mask)]

    X_train = torch.tensor(X_train)
    Y_train = torch.tensor(Y_train)
    X_val = torch.tensor(X_val)
    Y_val = torch.tensor(Y_val)
    train_adj_matrix = torch.tensor(train_adj_matrix)
    val_adj_matrix = torch.tensor(val_adj_matrix)

    train(model, X_train, Y_train, train_adj_matrix, optimizer, criterion, 1000)

    _, accuracy, _, _, _, _ = test(model, X_val, Y_val, val_adj_matrix, criterion)
    print(f"Fold {fold} - Score: {accuracy:.4f}")

    np.append(scores, accuracy)


score = scores.mean(scores)
"""



"""
# Fase di Test

test(model(best_params), X_test, Y_test)
"""

