
def train(model, X, Y, edge_index, edge_weight, mask, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        preds = model.forward(X, edge_index, edge_weight)
        loss = criterion(preds[mask], Y[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
