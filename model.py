import torch
from torch_geometric.nn import GCNConv
from torch.nn import ReLU, Linear, Sigmoid
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, input_dim):
        super(GCN, self).__init__()

        # nella GCN la convoluzione è il meccanismo usato per implementare il message passing.
        # (VALUTARE: ChebConv | numero di conv layers da usare | dimensione hidd. layers)
        # self.conv = ChebConv(input_dim, hidden_dim, K=3, normalization="sym")

        hidden_dim = 12     # (DA VALUTARE)

        self.conv1 = GCNConv(input_dim, 12, normalize=True)     # convolutional layer 1
        self.conv2 = GCNConv(12, 6, normalize=True)    # convolutional layer 2
        self.conv3 = GCNConv(6, 3, normalize=True)
        self.linear = Linear(3, 1)                  # linear layer
        self.sigmoid = Sigmoid()
        self.relu = ReLU()                                              # ReLU (VALUTARE: altre activation functions)


    def forward(self, X, edge_index, edge_weight):
        X = self.conv1(X, edge_index, edge_weight)
        X = self.relu(X)

        X = F.dropout(X, p=0.2, training=self.training)     # dropout (possiamo valutare altri valori di p)
        X = self.conv2(X, edge_index, edge_weight)
        X = self.relu(X)

        X = F.dropout(X, p=0.2, training=self.training)  # dropout (possiamo valutare altri valori di p)
        X = self.conv3(X, edge_index, edge_weight)
        X = self.relu(X)

        X = self.linear(X)
        X = self.sigmoid(X).squeeze(1)     # per avere le preds in riga

        return X

