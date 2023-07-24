#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include"../../include/utils.h"

int main(int argc, char const *argv[]) {

    int node, rows, cols, rimanenti, max = 0;
    int *matrix, *distance_matrix;
    double *closeness_centrality;
    double time;

    clock_t begin, end;

    /* Input: numero di nodi del grafo */
    printf("Number of nodes: ");
    scanf("%d", &node);
    rows = node;
    cols = node;

    /* Allochiamo le strutture dati necessarie */
    matrix = (int*)malloc(rows * cols * sizeof(int));
    distance_matrix = (int*)malloc(rows * cols * sizeof(int));
    closeness_centrality = (double*)malloc(node * sizeof(double));

    /* Leggiamo la matrice di esempio da un file */
    readIMatrix(rows, cols, matrix, "data/sparse/8000/random_matrix.dat");

    /* Stampiamo la matrice di esempio */
    printIMatrix(rows, cols, matrix);

    begin = clock();

    /* Calcoliamo la distance matrix */
    distanceMatrix(rows, cols, matrix, distance_matrix);

    /* Calcoliamo la Closeness Centrality */
    for (int i = 0; i < node; i++) {
        int sum_dist = 0;
        for (int j = 0; j < cols; j++) {
            sum_dist += distance_matrix[i * cols + j];
        }
        closeness_centrality[i] = (double) node / sum_dist;
        if (closeness_centrality[i] > closeness_centrality[max]) {
            max = i;
        }
    }

    end = clock();

    /* Calcolo tempo di esecuzione */
    time = (double) (end - begin) / CLOCKS_PER_SEC;

    /* Stampa dei risultati */
    printf("\nCloseness Centrality\n");
    printf("\nmax: %d - score: %f\n", max+1, closeness_centrality[max]);
    printf("\ntime: %f ms\n\n", time * 1000);

    if (node <= 10) {
        for (int i = 0; i < node; i++) {
            printf("Score %d: %f\n", (i+1), closeness_centrality[i]);
        }
    }

    /* free della memoria */
    free(matrix);
    free(distance_matrix);
    free(closeness_centrality);
    
    return 0;
}