#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"../../include/utils.h"

int main(int argc, char const *argv[]) {
    
    int *r, *c;
    double *d_c;

    int n, r_size, c_size, max = 0;
    double time;

    clock_t begin, end;

    /* Input: numero di nodi e archi del grafo */
    printf("Number of nodes: ");
    scanf("%d", &n);
    r_size = n + 1;

    printf("Number of edges: ");
    scanf("%d", &c_size);
    c_size *= 2;
    
    /* Allocazione delle strutture dati */
    r = (int*)malloc(r_size * sizeof(int));
    c = (int*)malloc(c_size * sizeof(int));
    d_c = (double*)malloc(n * sizeof(double));

    readRCEgraph(r, c, r_size, c_size, "data/dense/16000/random_r.dat", "data/dense/16000/random_c.dat");

    begin = clock();

    /* Calcolo della degree-centrality */
    for(int i = 0; i < r_size-1; i++) {
        d_c[i] = (double) (r[i+1] - r[i]) / (n-1);
        if (d_c[i] > d_c[max]) {
            max = i;
        }
    }

    end = clock();

    /* Calcolo tempo di esecuzione */
    time = (double) (end - begin) / CLOCKS_PER_SEC;

    /* Stampa dei risultati */
    printf("\nDegree Centrality\n");
    printf("\nmax: %d - score: %f\n", max+1, d_c[max]);
    printf("\ntime: %f ms\n\n", time * 1000);

    if (n <= 10) {
        for(int i = 0; i < n; i++) {
            printf("Score %d: %f\n", i+1, d_c[i]);
        }
    }

    /* free della memoria */
    free(r);
    free(c);
    free(d_c);

    return 0;
}
