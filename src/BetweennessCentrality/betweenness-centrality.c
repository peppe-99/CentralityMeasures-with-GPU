#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"../../include/utils.h"

void count_shortest_paths(int src, int dst, int v, int node, int *visited, int *matrix, int shortest_path, int path_lenght, int *num_shortest_path_st, int *num_shortest_path_svt);

int main(int argc, char const *argv[]) {

    int node, rows, cols, num_shortest_path_svt, num_shortest_path_st, max = 0;
    int *matrix, *distance_matrix, *visited;
    double *betweenness_centrality;
    double time;

    clock_t begin, end;

    /* Input: numero di nodi del grafo */
    printf("Inserire numero di nodi: ");
    scanf("%d", &node);
    rows = node;
    cols = node;

    matrix = (int*)malloc(rows * cols * sizeof(int));
    distance_matrix = (int*)malloc(rows * cols * sizeof(int));
    betweenness_centrality = (double*)malloc(node * sizeof(int));
    visited = (int*)malloc(node * sizeof(int));

    /* Leggiamo la matrice di esempio da un file */
    readIMatrix(rows, cols, matrix, "data/random_matrix.dat");

    begin = clock();

    /* Calcoliamo la distance matrix */
    distanceMatrix(rows, cols, matrix, distance_matrix);

    /* Calcoliamo la Betweenness Centrality per ogni nodo */
    for (int v = 0; v < node; v++) {
        betweenness_centrality[v] = 0;
        /* Per ogni coppia di nodi s e t diversi da v */
        for (int s = 0; s < node; s++) {
            for (int t = s+1; t < node; t++) {
                if (s != v && v != t) {
                    printf("%d - %d\n", s+1, t+1);
                    num_shortest_path_st = 0;
                    num_shortest_path_svt = 0;

                    if (matrix[s * cols + t] == 1) {
                        num_shortest_path_st = 1;
                    } else {
                        count_shortest_paths(s, t, v, node, visited, matrix, distance_matrix[s * cols + t], 0, &num_shortest_path_st, &num_shortest_path_svt);
                    }
                    betweenness_centrality[v] += (double) num_shortest_path_svt / num_shortest_path_st;
                }
            }
        }
        if (betweenness_centrality[v] > betweenness_centrality[max]) {
            max = v;
        }
        break;
    }

    end = clock();

    /* Calcolo tempo di esecuzione */
    time = (double) (end - begin) / CLOCKS_PER_SEC;

    /* Stampa dei risultati */
    printf("\nBetweenness Centrality\n");
    printf("\nmax: %d - score: %f\n", max+1, betweenness_centrality[max]);
    printf("\ntime: %f ms\n\n", time * 1000);

    if (node <= 10) {
        for (int i  = 0; i < node; i++) {
            printf("Score %d: %f\n", i+1, betweenness_centrality[i]);
        }
    }

    /* free della memoria */
    free(matrix);
    free(distance_matrix);
    free(betweenness_centrality);

    return 0;
}

void count_shortest_paths(int src, int dst, int v, int node, int *visited, int *matrix, int shortest_path, int path_lenght, int *num_shortest_path_st, int *num_shortest_path_svt){
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
        for (int i = 0; i < node; i++) {
            if (matrix[src * node + i] && visited[i] != 1) {
                count_shortest_paths(i, dst, v, node, visited, matrix, shortest_path, path_lenght+1, num_shortest_path_st, num_shortest_path_svt);
            }
        }
        visited[src] = 0;
    }
}
