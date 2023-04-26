/* utils.h */
extern int readIMatrix(int rows, int cols, int *matrix, const char *filename);
extern int readDMatrix(int rows, int cols, double *matrix, const char *filename);
extern void swap(int **current_matrix, int **new_matrix);
extern void printIMatrix(int rows, int cols, int *matrix);
extern void printDMatrix(int rows, int cols, double *matrix);
extern void printVoidMatrix(int rows, int cols, void **matrix);
