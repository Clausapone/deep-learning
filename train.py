import numpy as np

# TRAINING FUNCTION
# training the model over epochs storing also the loss_history
# (forwarding over the whole dataset and computing the loss over the training set)
def train(model, X, Y, edge_index, edge_weight, training_mask, optimizer, criterion, epochs):

    loss_history = np.array([])
    for epoch in range(epochs):
        model.train()
        preds = model.forward(X, edge_index, edge_weight)
        loss = criterion(preds[training_mask], Y[training_mask])
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loss_history = np.append(loss_history, loss.item())

    return loss_history
