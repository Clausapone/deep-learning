def train(model, X, Y, edge_index, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        preds = model(X, edge_index)
        loss = criterion(preds, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
