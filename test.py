import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def test(model, X, Y, edge_index, edge_weight, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(X, edge_index, edge_weight)
        loss = criterion(preds, Y)

    acc = accuracy_score(preds.numpy(), Y.numpy())
    prec = precision_score(preds.numpy(), Y.numpy())
    rec = recall_score(preds.numpy(), Y.numpy())
    f1_s = f1_score(preds.numpy(), Y.numpy())
    conf = confusion_matrix(preds.numpy(), Y.numpy())

    return loss, acc, prec, rec, f1_s, conf     # da printare nel main

