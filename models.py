import torch
from torch_geometric.nn import GCNConv, ChebConv, GATConv
from torch.nn import ReLU, Linear, Sigmoid, LeakyReLU, ELU
import torch.nn.functional as F

torch.manual_seed(42)

# {GCN_Conv}
# GCNConv AS CONVOLUTIONAL LAYER, ReLU AS ACTIVATION FUNCTION
# (we used standard normalization)
class GCN_Conv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(GCN_Conv, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1, normalize=True)     # convolutional layer 1
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2, normalize=True)   # convolutional layer 2
        self.conv3 = GCNConv(hidden_dim2, hidden_dim3, normalize=True)   # convolutional layer 3
        self.linear = Linear(hidden_dim3, 1)                  # linear layer
        self.sigmoid = Sigmoid()                                         # result in terms of probability (Sigmoid is equivalent to Softmax in binary tasks)
        self.relu = ReLU()                                               # ReLU as activation function

    # forward propagation using 3 convolutional layers
    def forward(self, X, edge_index, edge_weight):
        X = self.conv1(X, edge_index, edge_weight)
        X = self.relu(X)

        X = self.conv2(X, edge_index, edge_weight)
        X = self.relu(X)

        X = self.conv3(X, edge_index, edge_weight)
        X = self.relu(X)

        X = self.linear(X)
        X = self.sigmoid(X).squeeze(1)

        return X


# {Cheb_Conv}
# ChebConv AS CONVOLUTIONAL LAYER, Leaky_ReLU AS ACTIVATION FUNCTION
# (we used the polynomial degree K=3 and the symmetric normalization)
class Cheb_Conv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Cheb_Conv, self).__init__()
        self.conv1 = ChebConv(input_dim, hidden_dim1, K=3, normalization="sym")         # convolutional layer 1
        self.conv2 = ChebConv(hidden_dim1, hidden_dim2, K=3, normalization="sym")       # convolutional layer 2
        self.conv3 = ChebConv(hidden_dim2, hidden_dim3, K=3, normalization="sym")       # convolutional layer 3
        self.linear = Linear(hidden_dim3, 1)                                 # linear layer
        self.sigmoid = Sigmoid()                                                        # result in terms of probability
        self.leaky_relu = LeakyReLU()                                                   # Leaky_ReLU as activation function

    def forward(self, X, edge_index, edge_weight):
        X = self.conv1(X, edge_index, edge_weight)
        X = self.leaky_relu(X)

        X = self.conv2(X, edge_index, edge_weight)
        X = self.leaky_relu(X)

        X = self.conv3(X, edge_index, edge_weight)
        X = self.leaky_relu(X)

        X = self.linear(X)
        X = self.sigmoid(X).squeeze(1)

        return X


# {GAT_Conv}
# GATConv AS CONVOLUTIONAL LAYER, ELU AS ACTIVATION FUNCTION
# (we used 2 attention heads and a dropout with p=0.2 to decrease overfitting)
class GAT_Conv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(GAT_Conv, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim1, dropout=0.2, heads=2)             # convolutional layer 1
        self.conv2 = GATConv(hidden_dim1, hidden_dim2, dropout=0.2, heads=2)           # convolutional layer 2
        self.conv3 = GATConv(hidden_dim2, hidden_dim3, dropout=0.2, heads=2)           # convolutional layer 3
        self.linear = Linear(hidden_dim3, 1)                                # linear layer
        self.sigmoid = Sigmoid()                                                       # result in terms of probability
        self.elu = ELU()                                                               # ELU as activation function

    def forward(self, X, edge_index, edge_weight):
        X = self.conv1(X, edge_index, edge_weight)
        X = self.elu(X)

        X = self.conv2(X, edge_index, edge_weight)
        X = self.elu(X)

        X = self.conv3(X, edge_index, edge_weight)
        X = self.elu(X)

        X = self.linear(X)
        X = self.sigmoid(X).squeeze(1)

        return X
