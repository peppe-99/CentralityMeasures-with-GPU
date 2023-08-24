#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include"../../include/utils.h"

int main(int argc, char const *argv[]) {
    
    int nodes, rows, cols, accuracy = 5, max = 0;
    double time;

    double factorial = 1, org_inv_temp = 1.0, inv_temp = 1.0, beta, sum = 0.0;
    double *matrix, *tmp_matrix, *pwd_matrix, *exp_matrix, *subgraph_centrality, *total_communicability;

    clock_t begin, end;

    /* Input: nodi del grafo */
    printf("Number of nodes: ");
    scanf("%d", &nodes);
    rows = nodes;
    cols = nodes;

    /* Allochiamo le strutture dati necessarie */
    matrix = (double*)malloc(rows * cols * sizeof(double));
    tmp_matrix = (double*)malloc(rows * cols * sizeof(double));
    pwd_matrix = (double*)malloc(rows * cols * sizeof(double));
    exp_matrix = (double*)malloc(rows * cols * sizeof(double));
    subgraph_centrality = (double*)malloc(nodes * sizeof(double));
    total_communicability = (double*)malloc(nodes * sizeof(double));


    /* Leggiamo la matrice da un file */
    readDMatrix(rows, cols, matrix, "data/dense/4000/random_matrix.dat");

    /* Inizializziamo tmp_matrix = matrix*/
    memcpy(tmp_matrix, matrix, rows * cols * sizeof(double));

    /* Inizializziamo e^A = I */
    for (int i = 0; i < (rows * cols); i += (nodes + 1)) {
        exp_matrix[i] = 1.0;
    }

    begin = clock();

    /* Calcoliamo e^A = I + A */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            exp_matrix[i * cols + j] += matrix[i * cols + j];
        }
    }

    /* Calcoliamo l'esponenziale di matrice */
    for (int k = 2; k <= accuracy; k++) {
        
        /* Aggiorniamo il fattoriale */
        factorial *= k;

        /* Aggiorniamo l'inverse temperature */
        inv_temp *= org_inv_temp;

        /* Calcolo beta = inv_temp^k / k! */
        beta = (double) (inv_temp / factorial);

        /* A^k = A * A^(k-1) */
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                pwd_matrix[i * cols + j] = 0;
                for (int k = 0; k < cols; k++) {
                    pwd_matrix[i * cols + j] += tmp_matrix[i * cols + k] * matrix[k * cols + j]; 
                }
            }
        }

        /* Aggiorniamo tmp_matrix */
        memcpy(tmp_matrix, pwd_matrix, rows * cols * sizeof(double));

        /* e^A += beta * A^k = (A^k)/k! */
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                exp_matrix[i * cols + j] += beta * pwd_matrix[i * cols + j];
            }
        }

    }

    end = clock();

    /* Calcolo tempo di esecuzione */
    time = (double) (end - begin) / CLOCKS_PER_SEC;

    /* SC(i) = [e^A]_ii */
    for (int i = 0; i < nodes; i++) {
        sum += exp_matrix[i * cols + i];
        subgraph_centrality[i] = exp_matrix[i * cols + i];
        if (subgraph_centrality[i] > subgraph_centrality[max]) {
            max = i;
        }
    }

    printf("\nSubgraph Centrality:\n");
    printf("\nmax: %d - score: %f\n", max+1, subgraph_centrality[max]/sum);
    printf("\ntime: %f ms\n\n", time * 1000);

    for (int i = 0; i < nodes; i++) {
        subgraph_centrality[i] /= sum;
        if (nodes <= 10) {
            printf("Score %d: %f\n", (i+1), subgraph_centrality[i]);
        }
    }
/*
    /* TC(i) = sum_{j=1}^n [e^A]_ij
    printf("\nTotal Node Communicability:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            total_communicability[i] += exp_matrix[i * cols + j];
        }
        printf("Score %d: %f\n", (i+1), total_communicability[i]);
    }
*/
    return 0;
}