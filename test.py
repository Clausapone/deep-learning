import torch
from torchmetrics import Accuracy, Precision, Recall


def test(model, X, Y, edge_index, edge_weight, mask, criterion):
    accuracy = Accuracy(task="binary")
    precision = Precision(task="binary")
    recall = Recall(task="binary")

    model.eval()
    with torch.no_grad():
        preds = model.forward(X, edge_index, edge_weight)
        loss = criterion(preds[mask], Y[mask])
        preds = torch.round(preds)

        accuracy(preds, Y)
        precision(preds, Y)
        recall(preds, Y)

        # Ottenere il valore delle metriche
        accuracy = accuracy(preds, Y)
        precision = precision(preds, Y)
        recall = recall(preds, Y)

    return loss.item(), accuracy.item(), precision.item(), recall.item()
