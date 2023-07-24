#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include"../../include/utils.h"

__global__ void dc_kernel(int n, int *R, int *C, double *dc);

int main(int argc, char const *argv[]) {
    
    int n, r_size, c_size, max = 0;
    int *h_r, *h_c;
    int *d_r, *d_c;
    double *h_dc, *d_dc;

    float time;

    cudaEvent_t start, stop;

    /* Input: numero di nodi e archi del grafo */
    printf("Number of nodes: ");
    scanf("%d", &n);
    r_size = n + 1;

    printf("Number of edges: ");
    scanf("%d", &c_size);
    c_size *= 2;

    /* Allocazione strutture dati host */
    h_r = (int*)malloc(r_size * sizeof(int));
    h_c = (int*)malloc(c_size * sizeof(int));
    h_dc = (double*)malloc(n * sizeof(double));

    /* Allocazione strutture dati device */
    cudaMalloc((void **) &d_r, r_size * sizeof(int));
    cudaMalloc((void **) &d_c, c_size * sizeof(int));
    cudaMalloc((void **) &d_dc, n * sizeof(double));

    readRCEgraph(h_r, h_c, r_size, c_size, "data/demo/row_offsets.dat", "data/demo/column_indices.dat");

    /* Copia da host a device */
    cudaMemcpy(d_r, h_r, r_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, c_size * sizeof(int), cudaMemcpyHostToDevice);

    /* Configurazione del Kernel */
    dim3 blockDim(64);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

    /* Creiamo gli eventi per il tempo */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // tempo di inizio

    /* Invocazione del kernel */
    dc_kernel<<<gridDim, blockDim>>>(n, d_r, d_c, d_dc);

    /* Calcolo tempo di esecuzione */
    cudaEventRecord(stop); // tempo di fine
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    /* Copia dei risultati da device a host */
    cudaMemcpy(h_dc, d_dc, n * sizeof(double), cudaMemcpyDeviceToHost);

    /* Individuazione del nodo più centrale */
    for (int i = 0; i < n; i++) {
        if (h_dc[i] > h_dc[max]) {
            max = i;
        }
    }

    /* Stampa dei risultati */
    printf("\nDegree Centrality:\n");
    printf("\nmax: %d - score: %f\n", max+1, h_dc[max]);
    printf("\ntime: %f ms\n\n", time);

    if (n <= 10) {
        for (int i = 0; i < n; i++) {
            printf("Score %d: %f\n", i+1, h_dc[i]);
        }
    }

    /* free della memoria */
    free(h_r);
    free(h_c);
    free(h_dc);
    cudaFree(d_r);
    cudaFree(d_c);
    cudaFree(d_dc);

    return 0;
}

__global__ void dc_kernel(int n, int *R, int *C, double *dc) {
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);

    if (idx < n) {
        dc[idx] = (double) (R[idx+1] - R[idx]) / (n - 1);
    }
}