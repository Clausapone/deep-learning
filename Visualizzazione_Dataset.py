import networkx as nx

def read_gml_no_duplicates(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Crea un grafo vuoto
    G = nx.DiGraph()
    edges_seen = set()
    edge_mode = False
    source = None
    target = None

    for line in lines:
        if 'edge [' in line:
            edge_mode = True
        elif 'node [' in line:
            edge_mode = False
        if edge_mode and 'source' in line:
            source = line.split()[-1].strip().strip('"')
        if edge_mode and 'target' in line:
            target = line.split()[-1].strip().strip('"')
            if (source, target) not in edges_seen:
                G.add_edge(source, target)
                edges_seen.add((source, target))

    return G

# Percorso al file GML
percorso_file = './polblogs.gml'  # Sostituisci con il percorso del tuo file

# Leggere il grafo senza duplicati
grafo = read_gml_no_duplicates(percorso_file)


# Disegnare il grafo diretto
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))
pos = nx.spring_layout(grafo)  # Layout per la posizione dei nodi
nx.draw_networkx(grafo, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, arrows=True, arrowstyle='-|>', arrowsize=20)
plt.title("Visualizzazione del Grafo Diretto GML")
plt.show()
