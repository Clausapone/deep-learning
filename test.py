import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix


# {TEST FUNCTION}
# computing and exporting the results as evaluation metrics
def test(model, X, Y, edge_index, edge_weight, mask, criterion):

    # instantiating metrics functions
    accuracy = Accuracy(task="binary")
    precision = Precision(task="binary")
    recall = Recall(task="binary")
    f1_score = F1Score(task="binary")
    conf_mat = ConfusionMatrix(task="binary")

    # computing the predictions in evaluating mode of pytorch
    # (forwarding over the whole dataset and computing the loss over the test set)
    model.eval()
    with torch.no_grad():
        preds = model.forward(X, edge_index, edge_weight)
        loss = criterion(preds[mask], Y[mask])
        preds = torch.round(preds)

        accuracy(preds, Y)
        precision(preds, Y)
        recall(preds, Y)
        f1_score(preds, Y)
        conf_mat(preds, Y)

        # Ottenere il valore delle metriche
        accuracy = accuracy(preds, Y)
        precision = precision(preds, Y)
        recall = recall(preds, Y)
        f1_score = f1_score(preds, Y)
        conf_mat = conf_mat(preds, Y)

    return loss.item(), accuracy.item(), precision.item(), recall.item(), f1_score.item(), conf_mat
