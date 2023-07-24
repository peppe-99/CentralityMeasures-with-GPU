#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
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

int readRCEgraph(int *r, int *c, int r_size, int c_size, const char *r_path, const char *c_path) {
    
    FILE *R = fopen(r_path, "r");
    FILE *C = fopen(c_path, "r");

    if (R == NULL || C == NULL) {
        return 0;
    }

    for (int i = 0; i < r_size; i++) {
        fscanf(R, "%d\n", &r[i]);
    }

    for (int i = 0; i < c_size; i++) {
        fscanf(C, "%d\n", &c[i]);
    }

    fclose(R);
    fclose(C);

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
    if (rows <= 10 && cols <= 10) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%d\t", matrix[i * cols + j]);
            }
            printf("\n");
        }
    }
}

void printDMatrix(int rows, int cols, double *matrix) {
    if (rows <= 10 && cols <= 10) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%.f\t", matrix[i * cols + j]);
            }
            printf("\n");
        }
    }
}

void distanceMatrix(int rows, int cols, int *matrix, int *distances) {
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i != j && matrix[i * cols + j] == 0) {
                distances[i * cols + j] = INT_MAX;
            }
            else {
                distances[i * cols + j] = matrix[i * cols + j];
            }        
        }
    }

    for (int k = 0; k < rows; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (
                    distances[i * cols + k] != INT_MAX && 
                    distances[k * cols + j] != INT_MAX &&
                    distances[i * cols + k] + distances[k * cols + j] < distances[i * cols + j]
                )
                distances[i * cols + j] = distances[i * cols + k] + distances[k * cols + j];
            }
        }
    }
}