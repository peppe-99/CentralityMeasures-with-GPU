#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<time.h>
#include"../../include/utils.h"

int main(int argc, char const *argv[]) {
    
    int *r, *c, *distances;
    double *c_c;

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
    c_c = (double*)malloc(n * sizeof(double));
    distances = (int*)malloc(n * n * sizeof(int));

    readRCEgraph(r, c, r_size, c_size, "data/demo/row_offsets.dat", "data/demo/column_indices.dat");

    begin = clock();

    /* Calcoliamo la distance matrix */
    for (int i = 0; i < r_size; i++) {
        for (int j = r[i]; j < r[i+1]; j++) {
            distances[i * n + c[j]] = 1;
            distances[c[j] * n + i] = 1;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && distances[i * n + j] != 1) {
                distances[i * n + j] = INT_MAX;
            }
        }      
    }

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (
                    distances[i * n + k] != INT_MAX && 
                    distances[k * n + j] != INT_MAX &&
                    distances[i * n + k] + distances[k * n + j] < distances[i * n + j]
                )
                distances[i * n + j] = distances[i * n + k] + distances[k * n + j];
            }
        }
    }

    /* Calcoliamo la Closeness Centrality */
    for (int i = 0; i < n; i++) {
        int sum_dist = 0;
        for (int j = 0; j < n; j++) {
            sum_dist += distances[i * n + j];
        }
        c_c[i] = (double) n / sum_dist;
        if (c_c[i] > c_c[max]) {
            max = i;
        }
    }

    end = clock();

    /* Calcolo tempo di esecuzione */
    time = (double) (end - begin) / CLOCKS_PER_SEC;

    /* Stampa dei risultati */
    printf("\nCloseness Centrality\n");
    printf("\nmax: %d - score: %f\n", max+1, c_c[max]);
    printf("\ntime: %f ms\n\n", time * 1000);

    if (n <= 10) {
        for (int i = 0; i < n; i++) {
            printf("Score %d: %f\n", (i+1), c_c[i]);
        }
    }

    /* free della memoria */
    free(r);
    free(c);
    free(c_c);
    free(distances);

    return 0;
}
