#include <stdio.h>
#include <stdlib.h>
#include"../../lib/utils.h"

int main(int argc, char const *argv[]) {
    
    int *r, *c;
    double *d_c;

    int n, r_size, c_size;

    /* Input: numero di nodi e archi del grafo */
    printf("Number of nodes: ");
    scanf("%d", &n);
    r_size = n + 1;

    printf("Number of edges: ");
    scanf("%d", &c_size);
    
    /* Allocazione delle strutture dati */
    r = (int*)malloc(r_size * sizeof(int));
    c = (int*)malloc(c_size * sizeof(int));
    d_c = (double*)malloc(n * sizeof(int));

    readRCEgraph(r, c, r_size, c_size, "data/row_offsets.dat", "data/column_indices.dat");

    /* Calcolo e stampa della degree-centrality */
    printf("\nDegree Centrality:\n");
    for(int i = 0; i < r_size-1; i++) {
        d_c[i] = (double) (r[i+1] - r[i]) / (n-1);
        printf("Score %d: %f\n", i, d_c[i]);
    }

    /* free della memoria */
    free(r);
    free(c);
    free(d_c);

    return 0;
}
