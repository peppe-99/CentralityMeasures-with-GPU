#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include"../../include/utils.h"

int main(int argc, char const *argv[]) {

    int node, rows, cols, max = 0;
    double time;

    int *matrix;
    double *degree_centrality_vector;

    clock_t begin, end;

    printf("Number of nodes: ");
    scanf("%d", &node);
    rows = node;
    cols = node;

    /* Allochiamo le strutture dati necessarie */
    matrix = (int*)malloc(rows * cols * sizeof(int));
    degree_centrality_vector = (double*)malloc(node * sizeof(double));

    /* Leggiamo la matrice di esempio da un file */
    readIMatrix(rows, cols, matrix, "data/demo/matrix.dat");

    /* Stampiamo la matrice di esempio */
    printIMatrix(rows, cols, matrix);

    begin = clock();

    /* Calcoliamo la degree centrality di ogni nodo */
    for (int i = 0; i < node; i++) {
        int neighbors = cols;
        degree_centrality_vector[i] = 0;

        for (int j = 0; j < neighbors; j++) {
            degree_centrality_vector[i] += matrix[i * cols + j];
        }
        degree_centrality_vector[i] /= (double)(node-1);
        if (degree_centrality_vector[i] > degree_centrality_vector[max]) {
            max = i;
        }
    }

    end = clock();

    /* Calcolo tempo di esecuzione */
    time = (double) (end - begin) / CLOCKS_PER_SEC;

    /* Stampa dei risultati */
    printf("\nDeegre Centrality\n");
    printf("\nmax: %d - score: %f\n", max+1, degree_centrality_vector[max]);
    printf("\ntime: %f ms\n\n", time * 1000);

    if (node <= 10) {
        for (int i  = 0; i < node; i++) {
            printf("Score %d: %f\n", i+1, degree_centrality_vector[i]);
        }
    }
    
    /* free della memoria */
    free(matrix);
    free(degree_centrality_vector);

    return 0;
}