#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<cublas_v2.h>
#include"../../include/utils.h"

int main(int argc, char const *argv[]) {
    
    int rows, cols, nodes, accuracy = 10;
    double alpha = 1.0, beta = 1.0, factorial = 1, sum = 0.0, org_inv_temp = 1.0, inv_temp = 1.0;

    double *h_matrix, *h_exp_matrix, *h_sgc, *h_tc;
    double *d_matrix, *d_pwd_matrix, *d_exp_matrix, *d_sgc;

    cublasHandle_t handle;
    size_t byte_matrix, byte_vector;

    /* Input: nodi del grafo */
    printf("Number of nodes: ");
    scanf("%d", &nodes);
    rows = nodes;
    cols = nodes;

    /* Bytes strutture dati */
    byte_matrix = rows * cols * sizeof(double);
    byte_vector = nodes * sizeof(double);

    /* Allochiamo strutture dati host */
    h_matrix = (double*) malloc(byte_matrix);
    h_exp_matrix = (double*) malloc(byte_matrix);
    h_sgc = (double*) malloc(byte_vector);
    h_tc = (double*) malloc(byte_vector);

    /* Allochiamo strutture dati device */
    cudaMalloc((void **) &d_matrix, byte_matrix);
    cudaMalloc((void **) &d_pwd_matrix, byte_matrix);
    cudaMalloc((void **) &d_exp_matrix, byte_matrix);
    cudaMalloc((void **) &d_sgc, byte_vector);

    /* Leggiamo la matrice di esempio da un file */
    readDMatrix(rows, cols, h_matrix, "data/demo/matrix.dat");

    /* Inizializzo h_exp_matrix */
    for (int i = 0; i < (rows * cols); i += (nodes+1)) {
        h_exp_matrix[i] = 1.0;
    }

    /* Creiamo l'handle */
    cublasCreate(&handle);

    /* Settiamo le strutture dati device */
    cublasSetMatrix(rows, cols, sizeof(double), h_matrix, rows, d_matrix, rows);
    cublasSetMatrix(rows, cols, sizeof(double), h_matrix, rows, d_pwd_matrix, rows);
    cublasSetMatrix(rows, cols, sizeof(double), h_exp_matrix, rows, d_exp_matrix, rows);

    /* Inizialmente e^A = I + A */
    cublasDgeam(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        rows, cols, &alpha, d_exp_matrix, 
        rows, &inv_temp, d_matrix, 
        rows, d_exp_matrix, rows
    );

    for (int k = 2; k <= accuracy; k++) {
        /* Aggiorno il fattoriale */
        factorial *= k;

        /* Aggiorno l'inverse temperature */
        inv_temp *= org_inv_temp;

        /* A^k = A * A^(k-1) */
        cublasDgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            rows, cols, nodes, 
            &alpha, d_matrix, rows,
            d_pwd_matrix, rows, &beta, 
            d_pwd_matrix, rows
        );

        /* Calcolo beta = inv_temp^k / k! */
        beta = (double) (inv_temp / factorial);

        /* e^A += beta * A^k = (A^k)/k! */
        cublasDgeam(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            rows, cols, &alpha, d_exp_matrix, 
            rows, &beta, d_pwd_matrix, 
            rows, d_exp_matrix, rows
        );
    }

    cublasGetMatrix(rows, cols, sizeof(double), d_exp_matrix, rows, h_exp_matrix, rows);

    /* SC(i) = [e^A]_ii */
    printf("\nSubgraph Centrality:\n");
    for (int i = 0; i < nodes; i++) {
        sum += h_exp_matrix[i * cols + i];
        h_sgc[i] = h_exp_matrix[i * cols + i];
    }

    for (int i = 0; i < nodes; i++) {
        h_sgc[i] /= sum;
        printf("Score %d: %f\n", (i+1), h_sgc[i]);
    }

    /* NC(i,j) = [e^A]_ij */
    printf("\nCommunicability between nodes:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("Score (%d, %d): %f\n", (i+1), (j+1), h_exp_matrix[i * cols + j]);
        }
    }

    /* TC(i) = sum_{j=1}^n [e^A]_ij */
    printf("\nTotal Node Communicability:\n");
    for (int i = 0; i < rows; i++) {
        h_tc[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            h_tc[i] += h_exp_matrix[i * cols + j];
        }
        printf("Score %d: %f\n", (i+1), h_tc[i]);
    }

    /* Distruggo l'handle */
    cublasDestroy(handle);

    /* free della memoria */
    free(h_tc);
    free(h_sgc);
    free(h_matrix);
    free(h_exp_matrix);
    cudaFree(d_sgc);
    cudaFree(d_matrix);
    cudaFree(d_pwd_matrix);
    cudaFree(d_exp_matrix);

    return 0;
}
