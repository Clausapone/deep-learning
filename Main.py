import torch
from torch import optim
from torch.nn import BCELoss
from model import GCN
from train import train
from test import test
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from itertools import product


# LOAD DEI DATI
Y = np.load("Y.npy")
X = np.load("X.npy")
edge_index = torch.load("edge_index.pt")
edge_weight = torch.load("edge_weight.pt")

#-----------------------------------------------------
# TRAIN AND TEST MASKS
train_mask, test_mask = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=42, shuffle=True)

#-----------------------------------------------------
# NORMALIZZAZIONE
scaler = RobustScaler()
X = scaler.fit_transform(X)

#-----------------------------------------------------
# CROSS VALIDATION
criterion = BCELoss()

X = torch.tensor(X, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.float)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# parametri da testare
params_grid = {'lr': [0.01, 0.001], 'wd': [0.05, 0.005], 'hidden_dim1': [32, 28], 'hidden_dim2': [18, 10], 'hidden_dim3': [3]}

# Genero tutte le combinazioni di parametri
combinations = product(params_grid['lr'], params_grid['wd'], params_grid['hidden_dim1'], params_grid['hidden_dim2'], params_grid['hidden_dim3'])

# scores per ogni combinazione di parametri
params_scores = {'lr': [], 'wd': [], 'hidden_dim1': [], 'hidden_dim2': [], 'hidden_dim3': [], 'model': [], 'score': []}

for lr, wd, hd1, hd2, hd3 in combinations:
    model = GCN(X.shape[1], hidden_dim1=hd1, hidden_dim2=hd2, hidden_dim3=hd3)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)     # (VALUTARE: SGD, RMSProp, Adagrad,...)

    score = np.array([])
    for fold, (train_mask1, val_mask) in enumerate(kf.split(X[train_mask])):

        # training
        train_loss = train(model, X, Y, edge_index, edge_weight, train_mask1, optimizer, criterion, 1000)

        # validation
        _, val_accuracy, _, _ = test(model, X, Y, edge_index, edge_weight, val_mask, criterion)

        score = np.append(score, val_accuracy)

    params_combination_score = np.mean(score)

    params_scores['lr'].append(lr)
    params_scores['wd'].append(wd)
    params_scores['hidden_dim1'].append(hd1)
    params_scores['hidden_dim2'].append(hd2)
    params_scores['hidden_dim3'].append(hd3)
    params_scores['score'].append(params_combination_score)
    params_scores['model'].append(model)


best_index = np.argmax(params_scores['score'])
best_params = {'lr': params_scores['lr'][best_index], 'wd': params_scores['wd'][best_index], 'hidden_dim1': params_scores['hidden_dim1'][best_index],
               'hidden_dim2': params_scores['hidden_dim2'][best_index], 'hidden_dim3': params_scores['hidden_dim3'][best_index],
               'model': params_scores['model'][best_index]}

best_model = best_params['model']


"""#--------------------------------------------------------------
# (PROVVISORIO) TRAINING DEL MODELLO
model = GCN(X.shape[1], 32, 10, 3)
criterion = BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)     # (VALUTARE: SGD, RMSProp, Adagrad,...)

X = torch.tensor(X, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.float)

train_loss = train(model, X, Y, edge_index, edge_weight, train_mask, optimizer, criterion, 1000)
"""
#-----------------------------------------------------
# TESTING DEL MODELLO
test_loss, test_accuracy, test_precision, test_recall = test(model, X, Y, edge_index, edge_weight, test_mask, criterion)

print(f"loss: {test_loss}, accuracy: {test_accuracy}, precision: {test_precision}, recall: {test_recall}")
