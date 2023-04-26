#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include"../../lib/utils.h"

__global__ void degree_centrality_GPU(int *matrix, double *degree_centrality, int node);

int main(int argc, char const *argv[]) {

    int node, rows, cols;
    int *h_matrix, *d_matrix;
    double *h_degree_centrality, *d_degree_centrality;

    size_t int_byte_matrix, byte_vector;

    /* Input: nodi del grafo */
    printf("Inserisci numero di nodi: ");
    scanf("%d", &node);
    rows = node;
    cols = node;

    int_byte_matrix = rows * cols * sizeof(int);
    byte_vector = node * sizeof(double);

    /* Allocazione strutture dati host */
    h_matrix = (int*)malloc(int_byte_matrix);
    h_degree_centrality = (double*)malloc(byte_vector);

    /* Lettura della matrice da file */
    readMatrix(rows, cols, h_matrix, "data/matrix.dat");

    /* Stampa della matrice */
    printMatrix(rows, cols, h_matrix);

    /* Allocazione strutture dati device*/
    cudaMalloc((void **) &d_matrix, int_byte_matrix);
    cudaMalloc((void **) &d_degree_centrality, byte_vector);

    /* Copia host -> device */
    cudaMemcpy(d_matrix, h_matrix, int_byte_matrix, cudaMemcpyHostToDevice);
    cudaMemset(d_degree_centrality, 0, byte_vector);

    /* Configurazione del kernel */
    dim3 blockDim(32, 32);
    dim3 gridDim(
        (cols + blockDim.x - 1) / blockDim.x,
        (rows + blockDim.y - 1) / blockDim.y
    );
    printf("blockDim = (%d,%d)\n", blockDim.x, blockDim.y);
    printf("gridDim = (%d,%d)\n", gridDim.x, gridDim.y);

    /* Inovazione del Kernel */
    degree_centrality_GPU<<<gridDim, blockDim>>>(d_matrix, d_degree_centrality, node);

    /* Copia device -> host dei risultati */
    cudaMemcpy(h_degree_centrality, d_degree_centrality, byte_vector, cudaMemcpyDeviceToHost);

    /* Stampa dei risultati */
    printf("\nDeegre Centrality\n");
    for (int i  = 0; i < node; i++) {
        h_degree_centrality[i] /= (double) (node-1);
        printf("Node: %d\tScore: %f\n", i+1, h_degree_centrality[i]);
    }

    /* free della memoria */
    free(h_matrix);
    free(h_degree_centrality);
    cudaFree(d_matrix);
    cudaFree(d_degree_centrality);
    
    return 0;
}

__global__ void degree_centrality_GPU(int *matrix, double *degree_centrality, int node) {
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    int j = threadIdx.y + (blockDim.y * blockIdx.y);

    if (i == 0 && j < node) {
        for (int k = 0; k < node; k++){
            degree_centrality[j] += (double) matrix[j * node + k];
        }
    }
}