#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[]) {
    
    /* File che rappresentano il grafo in formato RCE */
    FILE *R = fopen("data/row_offsets.dat", "r");
    FILE *C = fopen("data/column_indices.dat", "r");

    int *r, *c;
    double *d_c;

    int n, r_size, c_size;

    /* Input: numero di nodi e archi del grafo */
    printf("Inserire numero di nodi: ");
    scanf("%d", &n);
    r_size = n + 1;

    printf("Inserire numero di archi: ");
    scanf("%d", &c_size);
    
    /* Allocazione delle strutture dati */
    r = (int*)malloc(r_size * sizeof(int));
    c = (int*)malloc(c_size * sizeof(int));
    d_c = (double*)malloc(n * sizeof(int));


    /* Leggo da file il columns indices ed il row offsets array */
    for (int i = 0; i < r_size; i++) {
        fscanf(R, "%d\n", &r[i]);
    }

    for (int i = 0; i < c_size; i++) {
        fscanf(C, "%d\n", &c[i]);
    }

    /* Calcolo e stampa della degree-centrality */
    printf("Degree Centrality:\n");
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
