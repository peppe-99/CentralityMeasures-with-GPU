#include<stdio.h>
#include<stdlib.h>
#include <stdbool.h>
#include<cuda.h>
#include"../../lib/utils.h"

__global__ void bc_kernel(int n, int *R, int *C, double *bc, int *sigma, int *distance, double *dependency);

int main(int argc, char const *argv[]) {
    
    int n, r_size, c_size;
    int *h_r, *h_c;
    int *dev_r, *dev_c, *dev_sigma, *dev_distance;

    double *h_bc, *dev_bc, *dev_dependecy;
    
    /* Input: numero di nodi e archi del grafo */
    printf("Number of nodes: ");
    scanf("%d", &n);
    r_size = n + 1;

    printf("Number of edges: ");
    scanf("%d", &c_size);

    /* allocazione strutture host */
    h_r = (int*)malloc(r_size * sizeof(int));
    h_c = (int*)malloc(c_size * sizeof(int));
    h_bc = (double*)malloc(n * sizeof(double));

    /* allocazione strutture device */
    cudaMalloc((void **) &dev_r, r_size * sizeof(int));
    cudaMalloc((void **) &dev_c, c_size * sizeof(int));
    cudaMalloc((void **) &dev_sigma, n * sizeof(int));
    cudaMalloc((void **) &dev_distance, n * sizeof(int));
    cudaMalloc((void **) &dev_bc, n * sizeof(double));
    cudaMalloc((void **) &dev_dependecy, n * sizeof(double));

    readRCEgraph(h_r, h_c, r_size, c_size, "data/row_offsets.dat", "data/column_indices.dat");

    /* copio gli array row_offests e column_indices sul device */
    cudaMemcpy(dev_r, h_r, r_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, h_c, c_size * sizeof(int), cudaMemcpyHostToDevice);

    /* configurazione del Kernel */
    dim3 blockDim(64);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

    bc_kernel<<<gridDim, blockDim>>>(n, dev_r, dev_c, dev_bc, dev_sigma, dev_distance, dev_dependecy);
    cudaDeviceSynchronize();

    cudaMemcpy(h_bc, dev_bc, n * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("Score %d: %f\n", i+1, h_bc[i]);
    }

    /* free della memoria */
    free(h_r);
    free(h_c);
    free(h_bc);
    cudaFree(dev_r);
    cudaFree(dev_c);
    cudaFree(dev_bc);
    cudaFree(dev_sigma);
    cudaFree(dev_distance);
    cudaFree(dev_dependecy);

    return 0;
}

__global__ void bc_kernel(int n, int *R, int *C, double *bc, int *sigma, int *distance, double *dependency) {
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
                    for (int r = R[v]; r < R[v+1]; r++) {
                        int w = C[r];
                        if (distance[w] == INT_MAX) {
                            distance[w] = distance[v] + 1;
                            done = false;
                        }
                        if (distance[w] == (distance[v] + 1)) {
                            atomicAdd(&sigma[w], sigma[v]);
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
                    for (int r = R[v]; r < R[v+1]; r++) {
                        int w = C[r];
                        if (distance[w] == (distance[v] + 1)) {
                            if (sigma[w] != 0) {
                                dependency[v] += (sigma[v] * 1.0 / sigma[w]) * (1 + dependency[w]);
                            }
                        }
                    }
                    if (v != s) {
                        bc[v] += dependency[v] / 2;
                        /* divido per 2 perché ogni shortest path è stato contato due volte */
                    }
                }
            }
            __syncthreads();
        }
    }
}