import torch
from torch import optim
from torch.nn import BCELoss
from model import GCN
from train import train
from test import test
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import numpy as np

# LOAD DEI DATI
Y = np.load("Y.npy")
X = np.load("X.npy")
edge_index = torch.load("edge_index.pt")
edge_weight = torch.load("edge_weight.pt")


#--------------------------------------------------------------
# Train and test split per X, Y, edge_index, edge_weight
train_mask, test_mask = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=42, shuffle=True)

X_train, X_test = X[train_mask], X[test_mask]
Y_train, Y_test = Y[train_mask], Y[test_mask]

# Train_test_split per edge_index ed edge weight con maschera binaria
# OSSERVAZIONE: Un elemento ad un dato indice è True solo se source e target sono entrambi nel training set (o test set)
edges_training_mask = torch.tensor([(source.item() in train_mask and target.item() in train_mask) for source, target in zip(edge_index[0], edge_index[1])])
edges_test_mask = torch.tensor([(source.item() in test_mask and target.item() in test_mask) for source, target in zip(edge_index[0], edge_index[1])])

# Applico le maschere ad edge_index ed edge_weight
edge_index_train = edge_index[:, edges_training_mask]
edge_weight_train = edge_weight[edges_training_mask]

edge_index_test = edge_index[:, edges_test_mask]
edge_weight_test = edge_weight[edges_test_mask]

# mapping degli indici: gli indici dei nodi sono cambiati dopo lo split, quindi serve mappare i valori in edge_index a nuovi indici
map_train = {old_index: new_index for new_index, old_index in enumerate(train_mask)}
map_test = {old_index: new_index for new_index, old_index in enumerate(test_mask)}

# applicazione della mappa per train e test set
for old_index, new_index in map_train.items():
    edge_index_train[edge_index_train == old_index] = new_index
for old_index, new_index in map_test.items():
    edge_index_test[edge_index_test == old_index] = new_index


#--------------------------------------------------------------
# NORMALIZZAZIONE
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#--------------------------------------------------------------
# TRAINING DEL MODELLO

model = GCN(X_train.shape[1])
criterion = BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)     # (VALUTARE: SGD, RMSProp, Adagrad,...)

X_train = torch.tensor(X_train, dtype=torch.float)
Y_train = torch.tensor(Y_train, dtype=torch.float)

loss, preds = train(model, X_train, Y_train, edge_index_train, edge_weight_train, optimizer, criterion, 1000)
print(loss, preds)

"""
#--------------------------------------------------------------
# Fase di Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#for combination in parametrers_combinations
scores = np.array([])
for fold, (train_mask, val_mask) in enumerate(kf.split(X_train)):
    X_train, X_val = X_train[train_mask], X_train[val_mask]
    Y_train, Y_val = Y_train[train_mask], Y_train[val_mask]

    train_edge_index = edge_index[np.ix_(train_mask, train_mask)]
    val_edge_index = edge_index[np.ix_(val_mask, val_mask)]

    X_train = torch.tensor(X_train)
    Y_train = torch.tensor(Y_train)
    X_val = torch.tensor(X_val)
    Y_val = torch.tensor(Y_val)
    train_edge_index = torch.tensor(train_edge_index)
    val_edge_index = torch.tensor(val_edge_index)

    train(model, X_train, Y_train, train_edge_index, optimizer, criterion, 1000)

    _, accuracy, _, _, _, _ = test(model, X_val, Y_val, val_edge_index, criterion)
    print(f"Fold {fold} - Score: {accuracy:.4f}")

    np.append(scores, accuracy)


score = scores.mean(scores)
"""


"""
----------------------------------------------------------------------------------------
# Fase di Test

test(model(best_params), X_test, Y_test)
"""

