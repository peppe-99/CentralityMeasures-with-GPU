 #include<stdio.h>
 #include<stdlib.h>
 #include<time.h>
 
 int main(int argc, char const *argv[]) {
    
    FILE *random_matrix = fopen("data/random_matrix.dat", "w"); 
    FILE *random_r = fopen("data/random_r.dat", "w");
    FILE *random_c = fopen("data/random_c.dat", "w");

    int nodes, edges, r_size, c_size;
    int x = 0, y = 0;
    int *matrix, *r, *c;

    /* Inputs: numero di nodi e archi */
    printf("Number of nodes: ");
    scanf("%d", &nodes);
    r_size = nodes + 1;

    printf("Number of edges (min. %d, max. %d): ", nodes - 1, nodes * (nodes - 1));
    scanf("%d", &edges);
    c_size = edges * 2;

    matrix = (int*)malloc(nodes * nodes * sizeof(int));
    r = (int*)malloc(r_size * sizeof(int));
    c = (int*)malloc(c_size * sizeof(int));

    /* Garantiamo la connettività del grafo */
    for (int i = 0; i < nodes; i++) {
        int j = (i + 1) % nodes;
        matrix[i * nodes + j] = 1;
        matrix[j * nodes + i] = 1;
        edges--;
    }

    /* Generiamo casualmente i restanti archi */
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

    /* Creiamo i vettori r ed c per il formato CSR */
    int neighbors = 0;
    int num_edges = 0;
    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            if (matrix[i * nodes + j] == 1) {
                printf("%d\n", neighbors % c_size);
                c[neighbors % c_size] = j;
                neighbors++;
            }
        }
        r[i+1] = neighbors;
    }

    /* Creiamo il file con la matrice di adiacenza */
    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            fprintf(random_matrix, "%d ", matrix[i * nodes + j]);
        }
        fprintf(random_matrix, "\n");
    }

    /* Creiamo i file con i vettori del formato CSR */
    for (int i = 0; i < r_size; i++) {
        fprintf(random_r, "%d\n", r[i]);
    }

    for (int i = 0; i < c_size; i++) {
        fprintf(random_c, "%d\n", c[i]);
    }

    /* Chiudiamo gli stream */
    fclose(random_matrix);
    fclose(random_r);
    fclose(random_c);

    /* free della memoria */
    free(matrix);
    free(r);
    free(c);

    return 0;
 }
 