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


# DATA LOADING
# loading training data produced in data_storing.

Y = np.load("Y.npy")
X = np.load("X.npy")
edge_index = torch.load("edge_index.pt")
edge_weight = torch.load("edge_weight.pt")

#-----------------------------------------------------
# TRAIN AND TEST MASKS
# creating masks in order to split the dataset in training set and test set.

train_mask, test_mask = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=42, shuffle=True)

#-----------------------------------------------------
# NORMALIZATION
# input scaling by using the median and the interquartile range (IQR): less influenced by  outliers compared to standard scaler.

scaler = RobustScaler()
X = scaler.fit_transform(X)

#-----------------------------------------------------
# CROSS VALIDATION
# searching and training the best model between all possible models iterating over hyperparameters.

# setting loss criterion: Binary Cross Entropy loss for our binary classification task.
criterion = BCELoss()

X = torch.tensor(X, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.float)

# dictionary with all hyperparameters we will test in the validation phase.
params_grid = {'lr': [0.001, 0.0001], 'wd': [5e-4, 5e-5], 'hidden_dim1': [34, 32], 'hidden_dim2': [10, 8], 'hidden_dim3': [3, 2]}

# generating all possible combinations of parameters.
combinations = product(params_grid['lr'], params_grid['wd'], params_grid['hidden_dim1'], params_grid['hidden_dim2'], params_grid['hidden_dim3'])

# instantiating an empty dictionary: for a specific index i, each parameter-array, will contain the value of that hyperparameter
# for the i-th combination (and the associated Score), then we store also the i-th trained model.
params_scores = {'lr': [], 'wd': [], 'hidden_dim1': [], 'hidden_dim2': [], 'hidden_dim3': [], 'model': [], 'score': []}

# instantiating a function in order to compute indices for train and validation sets for K-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# iterating over hyperparameters
for lr, wd, hd1, hd2, hd3 in combinations:
    model = GCN(X.shape[1], hidden_dim1=hd1, hidden_dim2=hd2, hidden_dim3=hd3)      # instantiating the model with current combination of hyperparams
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)      # instantiating the optimizer with current combination of hyperparams

    score = np.array([])    # instantiating the array with the scores of each fold

    # iterating over folds
    for fold, (train_mask1, val_mask) in enumerate(kf.split(X[train_mask])):

        # training for i-th fold
        train(model, X, Y, edge_index, edge_weight, train_mask1, optimizer, criterion, 1000)

        # validation for i-th fold
        _, val_accuracy, _, _ = test(model, X, Y, edge_index, edge_weight, val_mask, criterion)

        score = np.append(score, val_accuracy)      # accumulator for scores of each fold (validation set accuracy)

    params_combination_score = np.mean(score)   # mean over scores in order to compute the params_combination score

    # updating params_scores
    params_scores['lr'].append(lr)
    params_scores['wd'].append(wd)
    params_scores['hidden_dim1'].append(hd1)
    params_scores['hidden_dim2'].append(hd2)
    params_scores['hidden_dim3'].append(hd3)
    params_scores['score'].append(params_combination_score)

i = np.argmax(params_scores['score'])   # index for best combination of parameters

# storing the best hyperparameters in a dictionary
best_params = {'lr': params_scores['lr'][i], 'wd': params_scores['wd'][i], 'hidden_dim1': params_scores['hidden_dim1'][i],
               'hidden_dim2': params_scores['hidden_dim2'][i], 'hidden_dim3': params_scores['hidden_dim3'][i]}

# instantiating best model and best optimizer
best_model = GCN(X.shape[1], hidden_dim1=best_params['hidden_dim1'], hidden_dim2=best_params['hidden_dim2'], hidden_dim3=best_params['hidden_dim3'])
best_optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'], weight_decay=best_params['wd'])

#--------------------------------------------------------------
# TRAINING
# final training with best_model and best_optimizer

train(best_model, X, Y, edge_index, edge_weight, train_mask, best_optimizer, criterion, 1000)

#-----------------------------------------------------
# EVALUATION
# testing the model and finally evaluating the metrics

test_loss, test_accuracy, test_precision, test_recall = test(best_model, X, Y, edge_index, edge_weight, test_mask, criterion)

print(f"test loss: {test_loss}, accuracy: {test_accuracy}, precision: {test_precision}, recall: {test_recall}")
