#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include"../../lib/utils.h"

void count_shortest_paths(int src, int dst, int v, int n, int *visited, int *r, int *c, int shortest_path, int path_lenght, int *num_shortest_path_st, int *num_shortest_path_svt);

int main(int argc, char const *argv[]) {

    int n, r_size, c_size, num_shortest_path_svt, num_shortest_path_st;
    
    int *r, *c, *distances, *visited;

    double *b_c;

    /* Input: numero di nodi e archi del grafo */
    printf("Number of nodes: ");
    scanf("%d", &n);
    r_size = n + 1;

    printf("Number of edges: ");
    scanf("%d", &c_size);

    /* allocazione strutture dati */
    r = (int*)malloc(r_size * sizeof(int));
    c = (int*)malloc(c_size * sizeof(int));
    visited = (int*)malloc(n * sizeof(int));
    b_c = (double*)malloc(n * sizeof(double));
    distances = (int*)malloc(n * n * sizeof(int));

    readRCEgraph(r, c, r_size, c_size, "data/row_offsets.dat", "data/column_indices.dat");

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

    for (int v = 0; v < n; v++) {
        b_c[v] = 0;
        /* Per ogni coppia di nodi s e t diversi da v */
        for (int s = 0; s < n; s++) {
            for (int t = s+1; t < n; t++) {
                if (s != v && v != t) {
                    num_shortest_path_st = 0;
                    num_shortest_path_svt = 0;

                    for (int i = r[s]; i < r[s+1]; i++) {
                        if (c[i] == t) {
                            num_shortest_path_st = 1;
                            break;
                        }
                    }

                    if (num_shortest_path_st == 0) {
                        count_shortest_paths(s, t, v, n, visited, r, c, distances[s * n + t], 0, &num_shortest_path_st, &num_shortest_path_svt);
                    }

                    b_c[v] += (double) num_shortest_path_svt / num_shortest_path_st;
                }
            }
        }
        printf("Score %d: %f\n", v+1, b_c[v]);
    }


    return 0;
}

void count_shortest_paths(int src, int dst, int v, int n, int *visited, int *r, int *c, int shortest_path, int path_lenght, int *num_shortest_path_st, int *num_shortest_path_svt) {
    visited[src] = 1;

    /* Passo base */
    if (src == dst) {
        if (path_lenght == shortest_path) {
            (*num_shortest_path_st)++;
            if (visited[v]) {
                (*num_shortest_path_svt)++;
            }
        }
        visited[src] = 0;
    } 
    else {
        /* Passo ricorsivo */
        for (int i = r[src]; i < r[src+1]; i++) {
            if (visited[c[i]] != 1) {
                count_shortest_paths(c[i], dst, v, n, visited, r, c, shortest_path, path_lenght+1, num_shortest_path_st, num_shortest_path_svt);
            }
        }
        visited[src] = 0;
    }
}