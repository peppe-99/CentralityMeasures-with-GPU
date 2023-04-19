#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>

__global__ void degree_centrality_GPU(int *matrix, int *degree_centrality, int node);
int readMatrix(int rows, int cols, int *matrix, const char *filename);

int main(int argc, char const *argv[]) {

    int node, rows, cols;
    int *h_matrix, *h_degree_centrality;
    int *d_matrix, *d_degree_centrality;

    size_t byte_matrix, byte_vector;

    printf("Inserisci numero di nodi: ");
    scanf("%d", &node);
    rows = node;
    cols = node;

    byte_matrix = rows * cols * sizeof(int);
    byte_vector = node * sizeof(int);

    h_matrix = (int*)malloc(byte_matrix);
    h_degree_centrality = (int*)malloc(byte_vector);

    readMatrix(rows, cols, h_matrix, "matrix.dat");

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d\t", h_matrix[i * cols + j]);
        }
        printf("\n");
    }

    cudaMalloc((void **) &d_matrix, byte_matrix);
    cudaMalloc((void **) &d_degree_centrality, byte_vector);

    cudaMemcpy(d_matrix, h_matrix, byte_matrix, cudaMemcpyHostToDevice);
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
    cudaDeviceSynchronize();


    printf("\nDeegre Centrality\n");
    for (int i  = 0; i < node; i++) {
        printf("Node: %d\tScore: %d\n", i+1, h_degree_centrality[i]);
    }
    
    return 0;
}

int readMatrix(int rows, int cols, int *matrix, const char *filename) {
    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        return 0;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(file, "%d", &matrix[i * cols + j]);
        }
    }

    fclose(file);
    return 1;
}

__global__ void degree_centrality_GPU(int *matrix, int *degree_centrality, int node) {
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    int j = threadIdx.y + (blockDim.y * blockIdx.y);

    if (i == 0 && j < node) {
        printf("(%d,%d)\n", i,j);
        degree_centrality[j] = 0;
        int neighbors = node;
        for (int k = 0; k < neighbors; k++){
            degree_centrality[j] += matrix[j * node + k];
        }
    }
}
