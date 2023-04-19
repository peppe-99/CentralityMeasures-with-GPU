#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int readMatrix(int rows, int cols, int *matrix, const char *filename);

int main(int argc, char const *argv[]) {

    int node, rows, cols;
    int *matrix;
    double *centrality_matrix;

    printf("Inserire numero di nodi: ");
    scanf("%d", &node);
    rows = node;
    cols = node;

    // Allochiamo le strutture dati necessarie
    matrix = (int*)malloc(rows * cols * sizeof(int));
    centrality_matrix = (double*)malloc(rows * cols * sizeof(double));

    // Leggiamo la matrice di esempio da un file
    readMatrix(rows, cols, matrix, "matrix.dat");

    // Stampiamo la matrice di esempio
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d\t", matrix[i * cols + j]);
        }
        printf("\n");
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
