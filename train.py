def train(model, X, Y, adj_matrix, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        preds = model(X, adj_matrix)
        loss = criterion(preds, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
