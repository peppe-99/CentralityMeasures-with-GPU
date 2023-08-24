#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<cuda.h>
#include"../../include/utils.h"

__global__ void bc_kernel(int n, int *matrix, double *bc, int *sigma, int *distance, double *dependency);

int main(int argc, char const *argv[]) {
    
    int nodes, rows, cols;
    float time;
    
    int *h_matrix, *d_matrix, *d_distances, *d_sigma;
    
    double *h_bc, *d_bc, *d_dependency;

    cudaEvent_t start, stop;

    /* Input: nodi del grafo */
    printf("Number of nodes: ");
    scanf("%d", &nodes);
    rows = nodes;
    cols = nodes;

    /* Allocazione strutture dati host */
    h_matrix = (int*) malloc(rows * cols * sizeof(int));
    h_bc = (double*) malloc(nodes * sizeof(double));

    /* Allocazione strutture dati device */
    cudaMalloc((void **) &d_matrix, rows * cols * sizeof(int));
    cudaMalloc((void **) &d_distances, nodes * sizeof(int));
    cudaMalloc((void **) &d_sigma, nodes * sizeof(int));
    cudaMalloc((void **) &d_dependency, nodes * sizeof(double));
    cudaMalloc((void **) &d_bc, nodes * sizeof(double));

    readIMatrix(rows, cols, h_matrix, "data/demo/matrix.dat");

    /* Copia della matrice da host a device */
    cudaMemcpy(d_matrix, h_matrix, rows * cols * sizeof(int), cudaMemcpyHostToDevice);

    /* Configurazione del kernel */
    dim3 blockDim(64);
    dim3 gridDim((nodes + blockDim.x - 1) / blockDim.x);

    /* Creiamo gli eventi per il tempo */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // tempo di inizio

    /* Invocazione del kernel */
    bc_kernel<<<gridDim, blockDim>>>(nodes, d_matrix, d_bc, d_sigma, d_distances, d_dependency);
    cudaDeviceSynchronize();

    /* Calcolo tempo di esecuzione */
    cudaEventRecord(stop); // tempo di fine
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    /* Copia dei risultati */
    cudaMemcpy(h_bc, d_bc, nodes * sizeof(double), cudaMemcpyDeviceToHost);

    /* Stampa dei risultati */
    printf("\nBetweenness Centrality\n");
    printf("time: %f ms\n\n", time);

    if (nodes <= 10) {
        for (int i = 0; i < nodes; i++) {
            printf("Score %d: %f\n", i+1, h_bc[i]);
        }
    }

    /* free della memoria */
    free(h_bc);
    free(h_matrix);
    cudaFree(d_bc);
    cudaFree(d_sigma);
    cudaFree(d_matrix);
    cudaFree(d_distances);
    cudaFree(d_dependency);
    
    return 0;
}

__global__ void bc_kernel(int n, int *matrix, double *bc, int *sigma, int *distance, double *dependency) {

    int idx = threadIdx.x;

    if (idx >= n) return;

    /* variabili shared */
    __shared__ int s;
    __shared__ int current_depth;
    __shared__ bool done;

        if (idx == 0) {
        s = -1;
    }
    __syncthreads();

    /* iteriamo per ogni nodo radice s */
    while (s < n - 1) {
        if (idx == 0) {
            s++;
            done = false;
            current_depth = - 1;
        }
        __syncthreads();

        /* inizializziamo distance, dependency e sigma per ogni nodo */
        for (int v = idx; v < n; v += blockDim.x) {
            if (v == s) {
                distance[v] = 0;
                sigma[v] = 1;
            }
            else {
                distance[v] = INT_MAX; // "infinito"
                sigma[v] = 0;
            }
            dependency[v] = 0.0;
        }
        __syncthreads();

        /*  
            eseguo una BFS per calcolare gli shortest 
            path e la distanza da s a tutti gli altri nodi 
        */
        while (!done) {
            if (idx == 0) {
                current_depth++;
            }
            done = true;
            __syncthreads();

            for (int v = idx; v < n; v += blockDim.x) {
                if (distance[v] == current_depth) {
                    for (int j = 0; j < n; j++) {
                        if (matrix[v * n + j] == 1) {
                            if (distance[j] == INT_MAX) {
                                distance[j] = distance[v] + 1;
                                done = false;
                            }
                            if (distance[j] == (distance[v] + 1)) {
                                atomicAdd(&sigma[j], sigma[v]);
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        /* eseguo una reverse BFS per aggiornare le betweenness centrality */
        while (current_depth) {
            if (idx == 0) {
                current_depth--;
            }
            __syncthreads();

            for (int v = idx; v < n; v += blockDim.x) {
                if (distance[v] == current_depth) {
                    for (int j = 0; j < n; j++) {
                        if (
                            matrix[v * n + j] == 1 &&
                            distance[j] == (distance[v] + 1) &&
                            sigma[j] != 0
                        )
                        dependency[v] += (sigma[v] * 1.0 / sigma[j]) * (1 + dependency[j]);
                    }
                    if (v != s) {
                        bc[v] += dependency[v] / 2;
                    }
                }
            }
            __syncthreads();
        }
    }
}
