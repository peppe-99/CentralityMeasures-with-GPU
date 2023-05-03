#include<stdio.h>
#include<stdlib.h>
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

void distanceMatrix(int rows, int cols, int *matrix, int *distance_matrix) {

    int *tmp_matrix = (int*)malloc(rows * cols * sizeof(int));
    int *pwd_matrix = (int*)malloc(rows * cols * sizeof(int));
    int rimanenti = (rows * cols) - rows;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            distance_matrix[i * cols + j] = matrix[i * cols + j];
            tmp_matrix[i * cols + j] = matrix[i * cols + j];
            if (matrix[i * cols + j] != 0) rimanenti--;
        }
    }

    for (int pwd = 2; rimanenti != 0; pwd++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                pwd_matrix[i * cols + j] = 0;
                for (int k = 0; k < cols; k++) {
                    pwd_matrix[i * cols + j] += tmp_matrix[i * cols + k] * matrix[k * cols + j];
                }
                if (i != j && pwd_matrix[i * cols + j] != 0 && distance_matrix[i * cols + j] == 0) {
                    distance_matrix[i * cols + j] = pwd;
                    rimanenti--;
                }
            }
        }
        swap(&pwd_matrix, &tmp_matrix);
    }
    free(tmp_matrix);
    free(pwd_matrix);
}
