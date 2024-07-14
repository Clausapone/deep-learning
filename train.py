def train(model, X_train, Y_train, adj_matrix, optimizer, criterion):
    epochs = 1000
    for epoch in range(epochs):
        model.train()
        preds = model(X_train, adj_matrix)
        loss = criterion(preds, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
