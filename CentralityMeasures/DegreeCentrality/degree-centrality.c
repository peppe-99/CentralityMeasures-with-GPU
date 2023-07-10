#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include"../../lib/utils.h"

int main(int argc, char const *argv[]) {

    int node, rows, cols;
    int *matrix;
    double *degree_centrality_vector;

    printf("Number of nodes: ");
    scanf("%d", &node);
    rows = node;
    cols = node;

    /* Allochiamo le strutture dati necessarie */
    matrix = (int*)malloc(rows * cols * sizeof(int));
    degree_centrality_vector = (double*)malloc(node * sizeof(double));

    /* Leggiamo la matrice di esempio da un file */
    readIMatrix(rows, cols, matrix, "data/matrix.dat");

    /* Stampiamo la matrice di esempio */
    printIMatrix(rows, cols, matrix);

    /* Calcoliamo la degree cebntrality di ogni nodo */
    printf("\nDeegre Centrality\n");
    for (int i  = 0; i < node; i++) {
        int neighbors = cols;
        degree_centrality_vector[i] = 0;

        for (int j = 0; j < neighbors; j++) {
            degree_centrality_vector[i] += matrix[i * cols + j];
        }
        degree_centrality_vector[i] /= (double)(node-1);

        printf("Score %d: %f\n", i+1, degree_centrality_vector[i]);
    }
    
    /* free della memoria */
    free(matrix);
    free(degree_centrality_vector);

    return 0;
}