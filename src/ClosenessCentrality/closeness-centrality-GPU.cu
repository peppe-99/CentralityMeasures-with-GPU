#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include"../../include/utils.h"

__global__ void closeness_centrality_GPU(double *distance_matrix, double *closeness_centrality, int node);

int main(int argc, char const *argv[]) {
    
    int rows, cols, node, to_inizialize;
    double alpha = 1.0;
    double beta = 1.0;

    double *h_matrix, *h_distance_matrix, *h_pwd_matrix, *h_closeness_centralities;
    double *d_matrix, *d_distance_matrix, *d_pwd_matrix, *d_closeness_centralities;
    
    cublasHandle_t handle;
    size_t byte_matrix, byte_vector;

    /* Input: nodi del grafo */
    printf("Number of nodes: ");
    scanf("%d", &node);
    rows = node;
    cols = node;

    /* Celle della distance matrix da inizializzare */
    to_inizialize = (node * node) - node;

    /* Bytes strutture dati */
    byte_matrix = rows * cols * sizeof(double);
    byte_vector = node * sizeof(double);

    /* Allociamo strutture dati host */
    h_matrix = (double*) malloc(byte_matrix);
    h_pwd_matrix = (double*) malloc(byte_matrix);
    h_distance_matrix = (double*) malloc(byte_matrix);
    h_closeness_centralities = (double*) malloc(byte_vector);

    /* Allochiamo strutture dati device */
    cudaMalloc((void **) &d_matrix, byte_matrix);
    cudaMalloc((void **) &d_pwd_matrix, byte_matrix);
    cudaMalloc((void **) &d_distance_matrix, byte_matrix);
    cudaMalloc((void **) &d_closeness_centralities, byte_vector);

    /* Leggiamo la matrice di esempio da un file */
    readDMatrix(rows, cols, h_matrix, "data/demo/matrix.dat");
    printDMatrix(rows, cols, h_matrix);

    /* Copiamo le distance che valgono 1 */
    memcpy(h_distance_matrix, h_matrix, byte_matrix);

    /* Creiamo l'handle di cublas */
    cublasCreate(&handle);

    /* Settiamo le strutture dati device */
    cublasSetMatrix(rows, cols, sizeof(double), h_matrix, rows, d_matrix, rows);
    cublasSetMatrix(rows, cols, sizeof(double), h_matrix, rows, d_pwd_matrix, rows);

    
    /* Contiamo quante celle della distance matrix già sono valorizzate*/
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (h_distance_matrix[i * cols + j] != 0) to_inizialize--;
        }
    }

    /* Calcoliamo la distance matrix */
    for (int pwd = 2; to_inizialize != 0; pwd++) {
        /* Prodotto matrice-matrice per ottenere la potenza pwd-esima */
        cublasDgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            rows, cols, node, 
            &alpha, d_matrix, rows, 
            d_pwd_matrix, rows, &beta, 
            d_pwd_matrix, rows
        );
        cublasGetMatrix(rows, cols, sizeof(double), d_pwd_matrix, rows, h_pwd_matrix, rows);

        /* Valorizziamo nuove celle della distance matrix */
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < cols; j++) {
                if (i != j && h_pwd_matrix[i * cols + j] != 0 && h_distance_matrix[i * cols + j] == 0) {
                    h_distance_matrix[i * cols + j] = pwd;
                    to_inizialize--;
                }
            }
        }
    } 

    /* Stampiamo la distance matrix */
    printf("\nDistance matrix\n");
    printDMatrix(rows, cols, h_distance_matrix);

    /* Copiamo sul device la distance matrix */
    cudaMemcpy(d_distance_matrix, h_distance_matrix, byte_matrix, cudaMemcpyHostToDevice);

    /* Configurazione del kernel */
    dim3 blockDim(32, 32);
    dim3 gridDim(
        (cols + blockDim.x - 1) / blockDim.x,
        (rows + blockDim.y - 1) / blockDim.y
    );

    /* Invochiamo il kernel per sommare le righe*/
    closeness_centrality_GPU<<<gridDim, blockDim>>>(d_distance_matrix, d_closeness_centralities, node);

    cudaMemcpy(h_closeness_centralities, d_closeness_centralities, byte_vector, cudaMemcpyDeviceToHost);

    /* Stampa delle closeness centralities */
    printf("\nCloseness Centrality\n");
    for (int i = 0; i < node; i++) {
        printf("Score %d: %f\n", (i+1), h_closeness_centralities[i]);
    }

    /* Distruggo l'handle */
    cublasDestroy(handle);

    /* free della memoria */
    free(h_matrix);
    free(h_pwd_matrix);
    free(h_distance_matrix);
    free(h_closeness_centralities);
    cudaFree(d_matrix);
    cudaFree(d_pwd_matrix);
    cudaFree(d_distance_matrix);
    cudaFree(d_closeness_centralities);
   
    return 0;
}

__global__ void closeness_centrality_GPU(double *distance_matrix, double *closeness_centrality, int node) {
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    int j = threadIdx.y + (blockDim.y * blockIdx.y);

    if (i == 0 && j < node) {
        for (int k = 0; k < node; k++){
            closeness_centrality[j] += distance_matrix[j * node + k];
        }
        closeness_centrality[j] = (double) node / closeness_centrality[j];
    }
}
