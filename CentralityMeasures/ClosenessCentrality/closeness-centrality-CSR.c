#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include"../../lib/utils.h"

int main(int argc, char const *argv[]) {
    
    int *r, *c, *distances;
    double *c_c;

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
    c_c = (double*)malloc(n * sizeof(int));
    distances = (int*)malloc(n * n * sizeof(int));

    readRCEMatrix(r, c, r_size, c_size, "data/row_offsets.dat", "data/column_indices.dat");

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

    printIMatrix(n, n, distances);

    printf("\nCloseness Centrality:\n");
    for (int i = 0; i < n; i++) {
        int sum_dist = 0;
        for (int j = 0; j < n; j++) {
            sum_dist += distances[i * n + j];
        }
        c_c[i] = (double) n / sum_dist;
        printf("Score %d: %f\n", (i+1), c_c[i]);
    }

    return 0;
}
