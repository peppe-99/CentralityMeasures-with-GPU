import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

f = open("data/matrix.dat", "r")

adj_matrix = []
row_matrix = []

for row in f:
    for col in row:
        if (col == '0' or col == '1'):
            row_matrix.append(int(col))
        if (col == '\n' or col == ''):
            adj_matrix.append(row_matrix)
            row_matrix = []


adj_matrix = np.array(adj_matrix)
rows = len(adj_matrix)
cols = len(adj_matrix[0])

G = nx.Graph()

for i in range(rows):
    for j in range(cols):
        if (adj_matrix[i][j] == 1):
            G.add_edge(i+1, j+1)

nx.draw_kamada_kawai(G, with_labels=True)
plt.show()