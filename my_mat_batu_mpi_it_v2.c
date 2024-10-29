#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define ROWS 20
#define COLS 20

void generateRandomMatrix(int matrix[ROWS*COLS]) {
    int i, j;

    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            matrix[COLS*i+j] = rand() % 100;  // Random numbers between 0-99
        }
    }
}

void printMatrix(int matrix[ROWS*COLS]) {
    int i, j;

    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            printf("%3d ", matrix[COLS*i+j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {

    int it=0;
    int itmax=ROWS*COLS;

    int matrix1[ROWS*COLS];
    int matrix2[ROWS*COLS];
    int result[ROWS*COLS];

    int *recvcounts = NULL;
    int *displs = NULL;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        srand(time(NULL));  // Initiate random seed
        generateRandomMatrix(matrix1);
        generateRandomMatrix(matrix2);
    }

    MPI_Bcast(matrix1, ROWS * COLS, MPI_INT, 0, MPI_COMM_WORLD);    //Broadcast both matrix
    MPI_Bcast(matrix2, ROWS * COLS, MPI_INT, 0, MPI_COMM_WORLD);

//    int start = ((ROWS*COLS) / size) * rank;
//    int end = start+(ROWS*COLS)/size;

    for (; it < itmax; it++)        //All needed iterations to calculate the matrix sum
    {
        if (it%size == rank)        //Divide workload in processes
        {
            result[it] = matrix1[it] + matrix2[it];
        }

    }
/*
    for (int i = start; i < end; i++) {
        result[i] = matrix1[i] + matrix2[i];
    }
*/
    int *buffer = NULL;
    if (rank == 0) {
        buffer = (int *)malloc(ROWS * COLS * sizeof(int));
    }

    recvcounts = (int*)malloc(size * sizeof(int));
    displs = (int*)malloc(size * sizeof(int));


    for(int i=1; i<size; i++){
        recvcounts[i] = 1;
        displs[i] = i;
    }

    MPI_Datatype intercalate_type;                                              //A type to gather intercalated
    MPI_Type_vector((ROWS*COLS)/size, 1, size, MPI_INT, &intercalate_type);
    MPI_Type_commit(&intercalate_type);

    if (rank!=0)
    {
        MPI_Send(&result[rank], 1, intercalate_type, 0, 0, MPI_COMM_WORLD);
    } else{
        for (int i = 1; i < size; i++)
        {
            MPI_Recv(&buffer[i], 1, intercalate_type, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    if (rank == 0) {
        printf("Matrix 1:\n");
        printMatrix(matrix1);

        printf("\nMatrix 2:\n");
        printMatrix(matrix2);

        printf("\nSum result:\n");
        printMatrix(buffer);

        free(buffer);
    }

    MPI_Finalize();
    return 0;
}
