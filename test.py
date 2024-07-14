import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def test(model, X_test, Y_test, adj_matrix, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(X_test, adj_matrix)
        loss = criterion(preds, Y_test)

        acc = accuracy_score(preds.numpy(), Y_test.numpy())
        prec = precision_score(preds.numpy(), Y_test.numpy())
        rec = recall_score(preds.numpy(), Y_test.numpy())
        f1_s = f1_score(preds.numpy(), Y_test.numpy())
        conf = confusion_matrix(preds.numpy(), Y_test.numpy())

    return loss, acc, prec, rec, f1_s, conf     # da printare nel main

