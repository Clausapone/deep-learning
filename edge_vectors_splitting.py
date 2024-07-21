import torch

# Splitting per edge_index ed edge weight con maschera binaria per train_test_split e cross validation
def split(edge_index, edge_weight, train_mask, test_mask):

    # OSSERVAZIONE: Un elemento ad un dato indice è True solo se source e target sono entrambi nel training set (o test set)
    edges_training_mask = torch.tensor([(source.item() in train_mask and target.item() in train_mask) for source, target in zip(edge_index[0], edge_index[1])])
    edges_test_mask = torch.tensor([(source.item() in test_mask and target.item() in test_mask) for source, target in zip(edge_index[0], edge_index[1])])

    # Applico le maschere ad edge_index ed edge_weight
    edge_index_train = edge_index[:, edges_training_mask]
    edge_weight_train = edge_weight[edges_training_mask]

    edge_index_test = edge_index[:, edges_test_mask]
    edge_weight_test = edge_weight[edges_test_mask]

    # mapping degli indici: gli indici dei nodi sono cambiati dopo lo split, quindi serve mappare i valori in edge_index a nuovi indexes
    map_train = {old_index: new_index for new_index, old_index in enumerate(train_mask)}
    map_test = {old_index: new_index for new_index, old_index in enumerate(test_mask)}

    # applicazione della mappa per train e test set
    for old_index, new_index in map_train.items():
        edge_index_train[edge_index_train == old_index] = new_index
    for old_index, new_index in map_test.items():
        edge_index_test[edge_index_test == old_index] = new_index

    return edge_index_train, edge_weight_train, edge_index_test, edge_weight_test
