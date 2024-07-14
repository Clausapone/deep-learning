import torch
from torch_geometric.nn import GCNConv
from torch.nn import ReLU, Linear, Sigmoid
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, input_dim):
        super(GCN, self).__init__()

        # nella GCN la convoluzione è il meccanismo usato per implementare il message passing.
        #(VALUTARE: ChebConv | numero di conv layers da usare | dimensione hidd. layers)

        hidden_dim = 64     # (VALUTARE: 128, 32)

        self.conv1 = GCNConv(input_dim, hidden_dim, normalize=True)     # convolutional layer 1
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=True)    # convolutional layer 2
        self.conv3 = GCNConv(hidden_dim, hidden_dim, normalize=True)    # convolutional layer 3
        self.linear = Linear(hidden_dim, 1)                  # linear layer
        self.sigmoid = Sigmoid()
        self.relu = ReLU()                                              # ReLU (VALUTARE: altre activation functions)


    def forward(self, x, adj_matrix):
        x = self.conv1(x, adj_matrix)
        x = self.relu(x)

        x = F.dropout(x, p=0.2, training=self.training)     # dropout (possiamo valutare altri valori di p)
        x = self.conv2(x, adj_matrix)
        x = self.relu(x)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, adj_matrix)
        x = self.relu(x)

        x = self.linear(x)
        x = self.sigmoid(x)

        return x

