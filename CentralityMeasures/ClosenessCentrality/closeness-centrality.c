#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include"../../lib/utils.h"

int main(int argc, char const *argv[]) {

    int node, rows, cols, rimanenti;
    int *matrix, *distance_matrix;
    double *closeness_centrality;

    /* Input: numero di nodi del grafo */
    printf("Inserire numero di nodi: ");
    scanf("%d", &node);
    rows = node;
    cols = node;

    /* Allochiamo le strutture dati necessarie */
    matrix = (int*)malloc(rows * cols * sizeof(int));
    distance_matrix = (int*)malloc(rows * cols * sizeof(int));
    closeness_centrality = (double*)malloc(node * sizeof(double));

    /* Leggiamo la matrice di esempio da un file */
    readIMatrix(rows, cols, matrix, "data/matrix.dat");

    /* Stampiamo la matrice di esempio */
    printIMatrix(rows, cols, matrix);

    /* Calcoliamo la distance matrix */
    distanceMatrix(rows, cols, matrix, distance_matrix);

    /* Stampiamo la distance matrix */
    printf("\nDistance Matrix:\n");
    printIMatrix(rows, cols, distance_matrix);

    /* Calcoliamo la Closeness Centrality */
    printf("\nCloseness Centrality:\n");
    for (int i = 0; i < node; i++) {
        int sum_dist = 0;
        for (int j = 0; j < cols; j++) {
            sum_dist += distance_matrix[i * cols + j];
        }
        closeness_centrality[i] = (double) (node - 1) / sum_dist;
        printf("Node: %d\tScore: %f\n", (i+1), closeness_centrality[i]);
    }

    /* free della memoria */
    free(matrix);
    free(distance_matrix);
    free(closeness_centrality);
    
    return 0;
}