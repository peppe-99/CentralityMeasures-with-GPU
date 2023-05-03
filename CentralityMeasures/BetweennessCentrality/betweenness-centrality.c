#include<stdio.h>
#include<stdlib.h>
#include"../../lib/utils.h"

int main(int argc, char const *argv[]) {

    int node, rows, cols, num_shortest_path_svt, num_shortest_path_st;
    int *matrix, *distance_matrix;
    double *betweenness_centrality;

    /* Input: numero di nodi del grafo */
    printf("Inserire numero di nodi: ");
    scanf("%d", &node);
    rows = node;
    cols = node;

    matrix = (int*)malloc(rows * cols * sizeof(int));
    distance_matrix = (int*)malloc(rows * cols * sizeof(int));
    betweenness_centrality = (double*)malloc(node * sizeof(int));

    /* Leggiamo la matrice di esempio da un file */
    readIMatrix(rows, cols, matrix, "data/matrix.dat");

    /* Stampiamo la matrice di esempio */
    printf("Matrice di Adiacenza:\n");
    printIMatrix(rows, cols, matrix);

    /* Calcoliamo la distance matrix */
    distanceMatrix(rows, cols, matrix, distance_matrix);

    /* Stampiamo la distance matrix */
    printf("\nDistance Matrix\n");
    printIMatrix(rows, cols, distance_matrix);

    /* Calcoliamo la Betweenness Centrality per ogni nodo */
    for (int v = 0; v < node; v++) {
        betweenness_centrality[v] = 0;
        /* Per ogni coppia di nodi s e t diversi da v */
        for (int s = 0; s < node; s++) {
            for (int t = 0; t < node; t++) {
                if (s != v && v != t && s != t) {
                    num_shortest_path_st = 0;
                    num_shortest_path_svt = 0;

                    /* Contiamo gli shortest path tra s e t e quelli che contengono v */
                    for (int i = 0; i < node; i++) {
                        for (int j = 0; j < node; j++) {
                            if ((distance_matrix[s * cols + i] + distance_matrix[j * cols + t] + matrix[i * cols + j]) == distance_matrix[s * cols + t] ||
                                (distance_matrix[s * cols + j] + distance_matrix[i * cols + t] + matrix[j * cols + i]) == distance_matrix[s * cols + t]) {
                                num_shortest_path_st++;
                                if (i == v || j == v) num_shortest_path_svt++;
                            }
                        }
                    }
                    betweenness_centrality[v] += (double) num_shortest_path_svt / num_shortest_path_st;
                }
            }
        }        
        printf("Node: %d\tScore: %f\n", v+1, betweenness_centrality[v]);
    }

    /* free della memoria */
    free(matrix);
    free(distance_matrix);
    free(betweenness_centrality);

    return 0;
}
