import torch
from torch import optim
from torch.nn import BCELoss
from model import GCN
from train import train
from test import test
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from edge_splitting import edge_splitting

# LOAD DEI DATI
Y = np.load("Y.npy")
X = np.load("X.npy")
edge_index = torch.load("edge_index.pt")
edge_weight = torch.load("edge_weight.pt")


#-----------------------------------------------------
# Train and test split per X, Y, edge_index, edge_weight
train_mask, test_mask = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=42, shuffle=True)

X_train, X_test = X[train_mask, :], X[test_mask, :]
Y_train, Y_test = Y[train_mask], Y[test_mask]

edge_index_train, edge_weight_train, edge_index_test, edge_weight_test = edge_splitting(edge_index, edge_weight, train_mask, test_mask)

#-----------------------------------------------------
# NORMALIZZAZIONE
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#-----------------------------------------------------
# CROSS VALIDATION
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#for combination in parametrers_combinations

model = GCN(X_train.shape[1])
criterion = BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)     # (VALUTARE: SGD, RMSProp, Adagrad,...)

scores = np.array([])
for fold, (train_mask, val_mask) in enumerate(kf.split(X_train)):

    # train_validation_split

    X_train, X_val = X_train[train_mask], X_train[val_mask]
    Y_train, Y_val = Y_train[train_mask], Y_train[val_mask]

    X_train = torch.tensor(X_train, dtype=torch.float)
    Y_train = torch.tensor(Y_train, dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.float)
    Y_val = torch.tensor(Y_val, dtype=torch.float)

    # splitting di edge_index ed edge_weight in train e validation partendo da edge_index_train e
    # edge_weight_train ottenute in train_test_split
    edge_index_train, edge_weight_train, edge_index_val, edge_weight_val = edge_splitting(edge_index_train, edge_weight_train, train_mask, test_mask)

    # training
    train_loss, train_accuracy = train(model, X_train, Y_train, edge_index_train, edge_weight_train, optimizer, criterion, 1000)

    # validation
    val_loss, val_accuracy = test(model, X_val, Y_val, edge_index_val, edge_weight_val, criterion)

    np.append(scores, val_accuracy)

# params_grid['parameter_combination_score'] = scores.mean(scores)


# params_grid = {'lr': 0.01, 'decay': 0.002, 'parameter_combination_score': 0.75}

# best_params = ...

# best model = model(**best_params)


#-----------------------------------------------------
# TESTING DEL MODELLO
X_test = torch.tensor(X_test, dtype=torch.float)
Y_test = torch.tensor(Y_test, dtype=torch.float)

test_loss, accuracy = test(model, X_test, Y_test, edge_index_test, edge_weight_test, criterion)

print(f"loss: {test_loss}, accuracy: {accuracy}")




