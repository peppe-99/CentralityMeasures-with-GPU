/* utils.h */
extern int readIMatrix(int rows, int cols, int *matrix, const char *filename);
extern int readDMatrix(int rows, int cols, double *matrix, const char *filename);
extern void printIMatrix(int rows, int cols, int *matrix);
extern void printDMatrix(int rows, int cols, double *matrix);
extern void swap(int **current_matrix, int **new_matrix);
extern void distanceMatrix(int rows, int cols, int *matrix, int *distance_matrix);
