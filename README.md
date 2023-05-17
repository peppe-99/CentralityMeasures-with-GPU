# NetworkAnalysis-with-GPU
This repository is dedicated to my master's degree project. The main goal is to implement, using GPU parallelism, algorithms that compute the "_centrality_" of each node and the "_connectivity_" of the network. For now, I have implemented these algorithms:
- **Degree Centrality** (sequential & parallel version): in this case the "centrality" of a node depends on its degree;
- **Closeness Centrality** (sequential & parallel version): here the "centrality" of a node depends on the distances separating it from other nodes in the network;
- **Betweenness centrality** (sequential & parallel version): here the "centrality" of a node depends on the number of "_shortest paths_" it partecipates in.

For the first two algorithms, the graph is represented by an adjacency matrix, a $n\times n$ matrix where an element in position $i,j$ is equal to $1$ if an edge exists from $i$ to $j$. Viceversa, for the third algoritm, the graph is represented with the Compressed Sparse Row (CSR) format. This representation consists of two arrays: _column indices_ and _row offsets_. The column indices array is a concatenation of each vertex's adjacency list into an array of $m$ elements. Instead, the row offsets array is an $n+1$ element array that points at where each vertex's adjacency list begins and ends withis the column indices array.
