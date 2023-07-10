 #include<stdio.h>
 #include<stdlib.h>
 #include<time.h>
 
 int main(int argc, char const *argv[]) {
    
    int nodes, edges;
    int x = 0, y = 0;
    int *matrix;

    printf("Number of nodes: ");
    scanf("%d", &nodes);

    printf("Number of edges (min. %d, max. %d): ", nodes - 1, nodes * (nodes - 1));
    scanf("%d", &edges);

    matrix = (int*)malloc(nodes * nodes * sizeof(int));

    // garantiamo la connessione del grafo
    for (int i = 0; i < nodes; i++) {
        int j = (i + 1) % nodes;
        matrix[i * nodes + j] = 1;
        matrix[j * nodes + i] = 1;
        edges--;
    }

    // generiamo casualmente i restanti archi
    srand(time(NULL));

    while (edges > 0) {
        while (x == y || matrix[x * nodes + y] == 1) {
            x = rand() % nodes;
            y = rand() % nodes;
        }
        matrix[x * nodes + y] = 1;
        matrix[y * nodes + x] = 1;
        edges--;
    }

    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            fprintf(stderr, "%d ", matrix[i * nodes + j]);
        }
        fprintf(stderr, "\n");
    }
    
    return 0;
 }
 