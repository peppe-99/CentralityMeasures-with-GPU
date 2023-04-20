#include<stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include"../../lib/utils.h"

int main(int argc, char const *argv[]) {

    int node, rows, cols, rimanenti;
    int *matrix, *pwd_matrix, *tmp_matrix, *distance_matrix;
    double *closeness_centrality;

    /* Input: numero di nodi del grafo */
    printf("Inserire numero di nodi: ");
    scanf("%d", &node);
    rows = node;
    cols = node;

    /* Allochiamo le strutture dati necessarie */
    matrix = (int*)malloc(rows * cols * sizeof(int));
    tmp_matrix = (int*)malloc(rows * cols * sizeof(int));
    pwd_matrix = (int*)malloc(rows * cols * sizeof(int));
    distance_matrix = (int*)malloc(rows * cols * sizeof(int));
    closeness_centrality = (double*)malloc(node * sizeof(double));

    /* Leggiamo la matrice di esempio da un file */
    readMatrix(rows, cols, matrix, "data/matrix.dat");

    /* Stampiamo la matrice di esempio */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d\t", matrix[i * cols + j]);
        }
        printf("\n");
    }

    /* Calcoliamo la distance matrix */
    rimanenti = (node * node) - node; // numero di celle della distance matrix da valorizzare
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            distance_matrix[i * cols + j] = matrix[i * cols + j];
            tmp_matrix[i * cols + j] = matrix[i * cols + j];
            if (matrix[i * cols + j] != 0) rimanenti--;
        }
    }
    /* Calcoliamo le potenze della matrice di adiacenza finché non abbiamo ottenuto la distance matrix */
    for (int pwd = 2; rimanenti != 0; pwd++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                pwd_matrix[i * cols + j] = 0;
                for (int k = 0; k < cols; k++) {
                    pwd_matrix[i * cols + j] += tmp_matrix[i * cols + k] * matrix[k * cols + j];
                }
                if (i != j && pwd_matrix[i * cols + j] != 0 && distance_matrix[i * cols + j] == 0) {
                    distance_matrix[i * cols + j] = pwd;
                    rimanenti--;
                }
            }
        }
        swap(&pwd_matrix, &tmp_matrix);
    }

    /* Stampiamo la distance matrix */
    printf("\nDistance Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d\t", distance_matrix[i * cols + j]);
        }
        printf("\n");
    }

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
    
    return 0;
}