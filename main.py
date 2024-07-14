from utils import load_data
from torch import optim
from torch.nn import BCELoss
from model import GCN
from train import train
from test import test
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

# Funzione per ottenere le strutture dati per il training, dando in input il percorso del nostro database
file_path = 'toy_dataset_1.gml'
X, Y, adj_matrix = load_data(file_path)

# Train and test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalizzazione
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creazione modello, loss e optimizer
model = GCN(X_train.shape[1])
criterion = BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)     # (VALUTARE: SGD, RMSProp, Adagrad,...)

# Training


"""
# Cross-validation

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, valid_index) in enumerate(kf.split(X_train)):
"""
