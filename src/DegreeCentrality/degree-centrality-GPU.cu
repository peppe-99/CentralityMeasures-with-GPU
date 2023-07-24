#include<assert.h>
#include<stdio.h>
#include<cuda.h>
#include<time.h>
#include"../../include/utils.h"

__global__ void degree_centrality_GPU(int *matrix, double *degree_centrality, int node);

int main(int argc, char const *argv[]) {

    int node, rows, cols, max = 0;
    int *h_matrix, *d_matrix;
    double *h_degree_centrality, *d_degree_centrality;

    float time;

    size_t int_byte_matrix, byte_vector;

    cudaEvent_t start, stop;

    /* Input: nodi del grafo */
    printf("Number of nodes: ");
    scanf("%d", &node);
    rows = node;
    cols = node;

    int_byte_matrix = rows * cols * sizeof(int);
    byte_vector = node * sizeof(double);

    /* Allocazione strutture dati host */
    h_matrix = (int*)malloc(int_byte_matrix);
    h_degree_centrality = (double*)malloc(byte_vector);

    /* Lettura della matrice da file */
    readIMatrix(rows, cols, h_matrix, "data/demo/matrix.dat");

    /* Stampa della matrice */
    printIMatrix(rows, cols, h_matrix);

    /* Allocazione strutture dati device*/
    cudaMalloc((void **) &d_matrix, int_byte_matrix);
    cudaMalloc((void **) &d_degree_centrality, byte_vector);

    /* Copia host -> device */
    cudaMemcpy(d_matrix, h_matrix, int_byte_matrix, cudaMemcpyHostToDevice);
    cudaMemset(d_degree_centrality, 0, byte_vector);

    /* Configurazione del kernel */
    dim3 blockDim(8, 12);
    dim3 gridDim(
        (cols + blockDim.x - 1) / blockDim.x,
        (rows + blockDim.y - 1) / blockDim.y
    );

    /* Creiamo gli eventi per il tempo */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // tempo di inizio

    /* Inovazione del Kernel */
    degree_centrality_GPU<<<gridDim, blockDim>>>(d_matrix, d_degree_centrality, node);

    /* Calcolo tempo di esecuzione */
    cudaEventRecord(stop); // tempo di fine
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    /* Copia device -> host dei risultati */
    cudaMemcpy(h_degree_centrality, d_degree_centrality, byte_vector, cudaMemcpyDeviceToHost);

    /* Individuazione del nodo più centrale */
    for (int i = 0; i < node; i++) {
        if (h_degree_centrality[i] > h_degree_centrality[max]) {
            max = i;
        }
    }

    /* Stampa dei risultati */
    printf("\nDeegre Centrality\n");
    printf("\nmax: %d - score: %f\n", max+1, h_degree_centrality[max]);
    printf("\ntime: %f ms\n\n", time);
    
    if (node <= 10) {
    for (int i  = 0; i < node; i++) {
    for (int i  = 0; i < node; i++) {
        h_degree_centrality[i] /= (double) (node-1);
        for (int i  = 0; i < node; i++) {
        h_degree_centrality[i] /= (double) (node-1);
            printf("Score %d: %f\n", i+1, h_degree_centrality[i]);
        }
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
        degree_centrality[j] /= (double) (node - 1);
    }
}