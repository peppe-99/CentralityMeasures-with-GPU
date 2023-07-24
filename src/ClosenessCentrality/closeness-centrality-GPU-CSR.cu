#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include"../../include/utils.h"

__global__ void cc_kernel(int n, int *R, int *C, double *cc, int *distance);

int main(int argc, char const *argv[]) {
    
    int n, r_size, c_size, max = 0;
    int *h_r, *h_c;
    int *d_r, *d_c, *d_dist;
    double *h_cc, *d_cc;

    float time;

    cudaEvent_t start, stop;

    /* Input: numero di nodi e archi del grafo */
    printf("Number of nodes: ");
    scanf("%d", &n);
    r_size = n + 1;

    printf("Number of edges: ");
    scanf("%d", &c_size);
    c_size *= 2;

    /* allocazione strutture dati host */
    h_r = (int*)malloc(r_size * sizeof(int));
    h_c = (int*)malloc(c_size * sizeof(int));
    h_cc = (double*)malloc(n * sizeof(double));

    /* allocazione strutture dati device */
    cudaMalloc((void **) &d_r, r_size * sizeof(int));
    cudaMalloc((void **) &d_c, c_size * sizeof(int));
    cudaMalloc((void **) &d_dist, n * sizeof(int));
    cudaMalloc((void **) &d_cc, n * sizeof(double));

    readRCEgraph(h_r, h_c, r_size, c_size, "data/dense/16000/random_r.dat", "data/dense/16000/random_c.dat");

    /* copia gli array row_offests e column_indices sul device */
    cudaMemcpy(d_r, h_r, r_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, c_size * sizeof(int), cudaMemcpyHostToDevice);

    /* configurazione del Kernel */
    dim3 blockDim(1024);
    dim3 gridDim(1);
    
    /* Creiamo gli eventi per il tempo */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // tempo di inizio

    /* Invocazione del kernel */
    cc_kernel<<<gridDim, blockDim>>>(n, d_r, d_c, d_cc, d_dist);

    /* Calcolo tempo di esecuzione */
    cudaEventRecord(stop); // tempo di fine
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    /* Copia dei risultati da device a host */
    cudaMemcpy(h_cc, d_cc, n * sizeof(double), cudaMemcpyDeviceToHost);

    /* Individuazione del nodo più centrale */
    for (int i = 0; i < n; i++) {
        if (h_cc[i] > h_cc[max]) {
            max = i;
        }
    }

    /* Stampa dei risultati */
    printf("Closeness Centrality\n");
    printf("\nmax: %d - score: %f\n", max+1, h_cc[max]);
    printf("\ntime: %f ms\n\n", time);

    if (n <= 10) {
        for (int i = 0; i < n; i++) {
            printf("Score %d: %f\n", i+1, h_cc[i]);
        }
    }

    /* free della memoria */
    free(h_r);
    free(h_c);
    free(h_cc);
    cudaFree(d_r);
    cudaFree(d_c);
    cudaFree(d_cc);
    cudaFree(d_dist);
    
    return 0;
}

__global__ void cc_kernel(int n, int *R, int *C, double *cc, int *distance) {
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
        /* solo il thread 0 inizializza le variabili shared */
        if (idx == 0) {
            s++;
            done = false;
            current_depth = -1;
        }
        __syncthreads();

        /* inizializziamo le distanze */
        for (int v = idx; v < n; v += blockDim.x) {
            if (v == s) distance[v] = 0;
            else distance[v] = INT_MAX;
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
                    }
                }
            }
            __syncthreads();
        }
        
        if (idx == 0) {
            int dist_sum = 0;
            for (int i = 0; i < n; i++) {
                dist_sum += distance[i];
            }
            cc[s] = (double) n / dist_sum;
        }
        __syncthreads();
    }
}

