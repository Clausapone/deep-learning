import torch

def train(model, X, Y, edge_index, edge_weight, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        preds = model(X, edge_index, edge_weight)
        loss = criterion(preds, Y)
        preds = torch.round(preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return preds, loss.item()