#include<stdio.h>
#include"utils.h"

/* Funzioni per leggere una matrice da un file*/
int readIMatrix(int rows, int cols, int *matrix, const char *filename) {
    
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

int readDMatrix(int rows, int cols, double *matrix, const char *filename) {
    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        return 0;
    }

    int tmp;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(file, "%d", &tmp);
            matrix[i * cols + j] = (double) tmp;
        }
    }

    fclose(file);
    return 1;
}

/* Funzione per lo swap di matrici */
void swap(int **current_matrix, int **new_matrix) {
    int *temp = *current_matrix;
    *current_matrix = *new_matrix;
    *new_matrix = temp;
}

/* Funzioni per la stampa di una matrice */
void printIMatrix(int rows, int cols, int *matrix) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d\t", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void printDMatrix(int rows, int cols, double *matrix) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.f\t", matrix[i * cols + j]);
        }
        printf("\n");
    }
}