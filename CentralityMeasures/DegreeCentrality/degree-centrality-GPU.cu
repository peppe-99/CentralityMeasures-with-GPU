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

    size_t int_byte_matrix, double_byte_matrix, byte_vector;

    printf("Inserisci numero di nodi: ");
    scanf("%d", &node);
    rows = node;
    cols = node;

    int_byte_matrix = rows * cols * sizeof(int);
    double_byte_matrix = rows * cols * sizeof(double);
    byte_vector = node * sizeof(double);

    h_matrix = (int*)malloc(int_byte_matrix);
    h_degree_centrality = (double*)malloc(byte_vector);

    readMatrix(rows, cols, h_matrix, "data/matrix.dat");

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d\t", h_matrix[i * cols + j]);
        }
        printf("\n");
    }

    cudaMalloc((void **) &d_matrix, int_byte_matrix);
    cudaMalloc((void **) &d_degree_centrality, byte_vector);

    cudaMemcpy(d_matrix, h_matrix, int_byte_matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_degree_centrality, h_degree_centrality, byte_vector, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim(
        (cols + blockDim.x - 1) / blockDim.x,
        (rows + blockDim.y - 1) / blockDim.y
    );
    printf("blockDim = (%d,%d)\n", blockDim.x, blockDim.y);
    printf("gridDim = (%d,%d)\n", gridDim.x, gridDim.y);

    degree_centrality_GPU<<<gridDim, blockDim>>>(d_matrix, d_degree_centrality, node);

    cudaMemcpy(h_degree_centrality, d_degree_centrality, byte_vector, cudaMemcpyDeviceToHost);

    printf("\nDeegre Centrality\n");
    for (int i  = 0; i < node; i++) {
        h_degree_centrality[i] /= (double) (node-1);
        printf("Node: %d\tScore: %f\n", i+1, h_degree_centrality[i]);
    }
    
    return 0;
}

__global__ void degree_centrality_GPU(int *matrix, double *degree_centrality, int node) {
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    int j = threadIdx.y + (blockDim.y * blockIdx.y);

    if (i == 0 && j < node) {
        degree_centrality[j] = 0;
        int neighbors = node;
        for (int k = 0; k < neighbors; k++){
            degree_centrality[j] += (double) matrix[j * node + k];
        }
    }
}