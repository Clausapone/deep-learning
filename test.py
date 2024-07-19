import torch

def test(model, X, Y, edge_index, edge_weight, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(X, edge_index, edge_weight)
        loss = criterion(preds, Y)
        preds = torch.round(preds)
        correct = (preds == Y).sum()
        accuracy = int(correct) / len(Y)

    return loss, accuracy

