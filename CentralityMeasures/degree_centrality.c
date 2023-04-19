#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int readMatrix(int rows, int cols, int *matrix, const char *filename);

int main(int argc, char const *argv[]) {

    int node, rows, cols;
    int *matrix;
    int *degree_centrality_vector;

    printf("Inserire numero di nodi: ");
    scanf("%d", &node);
    rows = node;
    cols = node;

    // Allochiamo le strutture dati necessarie
    matrix = (int*)malloc(rows * cols * sizeof(int));
    degree_centrality_vector = (int*)malloc(node * sizeof(int));

    // Leggiamo la matrice di esempio da un file
    readMatrix(rows, cols, matrix, "matrix.dat");

    // Stampiamo la matrice di esempio
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d\t", matrix[i * cols + j]);
        }
        printf("\n");
    }

    // Calcoliamo la degree_centrality di ogni nodo
    printf("\nDeegre Centrality\n");
    for (int i  = 0; i < node; i++) {
        int neighbors = cols;
        degree_centrality_vector[i] = 0;

        for (int j = 0; j < neighbors; j++) {
            degree_centrality_vector[i] += matrix[i * cols + j];
        }

        printf("Node: %d\tScore: %d\n", i+1, degree_centrality_vector[i]);
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
