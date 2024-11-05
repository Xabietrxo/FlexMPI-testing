#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <empi.h>

#define ROWS 5
#define COLS 10

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

    int *buffer = NULL;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(ADM_COMM_WORLD, &rank);
    MPI_Comm_size(ADM_COMM_WORLD, &size);

    MPI_Datatype intercalate_type;                                  //A type to gather intercalated

    if (rank == 0) {
        srand(time(NULL));  // Initiate random seed
        generateRandomMatrix(matrix1);
        generateRandomMatrix(matrix2);
        buffer = (int *)malloc(ROWS * COLS * sizeof(int));
    }

    MPI_Bcast(matrix1, ROWS * COLS, MPI_INT, 0, ADM_COMM_WORLD);    //Broadcast both matrix
    MPI_Bcast(matrix2, ROWS * COLS, MPI_INT, 0, ADM_COMM_WORLD);

    // get process type
    int proctype;
    ADM_GetSysAttributesInt ("ADM_GLOBAL_PROCESS_TYPE", &proctype);

    // if process is native
    if (proctype == ADM_NATIVE) {
        
        printf ("Rank(%d/%d): Process native\n", rank, size);
        
    // if process is spawned
    } else {

        printf ("Rank(%d/%d): Process spawned\n", rank, size);
    }
    
    /* set max number of iterations */
    ADM_RegisterSysAttributesInt ("ADM_GLOBAL_MAX_ITERATION", &itmax);

    /* get actual iteration for new added processes*/
    ADM_GetSysAttributesInt ("ADM_GLOBAL_ITERATION", &it);

    /* starting monitoring service */
    ADM_MonitoringService (ADM_SERVICE_START);

    // init last world zize
    int last_size = size;

    // start loop
    for (; it < itmax; it++)        //All needed iterations to calculate the matrix sum
    {
        //Select hints on specific iterations
        int procs_hint = 0;
        int excl_nodes_hint = 0;
        if ( (it == 2*(itmax/10)) || (it == 4*(itmax/10)) ){
            procs_hint = 2;
            excl_nodes_hint = 0;
            ADM_RegisterSysAttributesInt ("ADM_GLOBAL_HINT_NUM_PROCESS", &procs_hint);
            ADM_RegisterSysAttributesInt ("ADM_GLOBAL_HINT_EXCL_NODES", &excl_nodes_hint);
        } else if ( (it == 6*(itmax/10)) || (it == 8*(itmax/10)) ){
            procs_hint = -2;
            excl_nodes_hint = 0;
            ADM_RegisterSysAttributesInt ("ADM_GLOBAL_HINT_NUM_PROCESS", &procs_hint);
            ADM_RegisterSysAttributesInt ("ADM_GLOBAL_HINT_EXCL_NODES", &excl_nodes_hint);
        }
        
        // update message if new spawned processes
        if (last_size != size) {
            
            MPI_Type_vector(it/size, 1, size, MPI_INT, &intercalate_type);
            MPI_Type_commit(&intercalate_type);

            if (rank==0)
            {
                for (int i = 1; i < size; i++)
                {
                    MPI_Recv(&buffer[i], 1, intercalate_type, i, 0, ADM_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            if (last_size<size)
            {
                MPI_Bcast(matrix1, ROWS * COLS, MPI_INT, 0, ADM_COMM_WORLD);    //Broadcast both matrix
                MPI_Bcast(matrix2, ROWS * COLS, MPI_INT, 0, ADM_COMM_WORLD);
            }

            last_size = size;
        }
        
        /* start malelability region */
        ADM_MalleableRegion (ADM_SERVICE_START);

        if (it%size == rank)        //Divide workload in processes
        {
            result[it] = matrix1[it] + matrix2[it];
        }

        printf("Rank (%d/%d): Iteration= %d\n", rank, size, it);

        // update the iteration value
        ADM_RegisterSysAttributesInt ("ADM_GLOBAL_ITERATION", &it);
        
        // ending malleable region
        int status;
        status = ADM_MalleableRegion (ADM_SERVICE_STOP);
        
        if (status == ADM_ACTIVE) {
            // updata rank and size
            MPI_Comm_rank(ADM_COMM_WORLD, &rank);
            MPI_Comm_size(ADM_COMM_WORLD, &size);
        } else {
            MPI_Send(&result[rank], 1, intercalate_type, 0, 0, ADM_COMM_WORLD);
            // end the process
            break;
        }
    }

    /* ending monitoring service */
    ADM_MonitoringService (ADM_SERVICE_STOP);

    
//!!!!------HAU ALDATZEKO-------!!!!------------------------(pentsatu nun jarri)------------------------------------------
/*    if (rank!=0)
    {
        MPI_Send(&result[rank], 1, intercalate_type, 0, 0, MPI_COMM_WORLD);
    } else{
        for (int i = 1; i < size; i++)
        {
            MPI_Recv(&buffer[i], 1, intercalate_type, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }*/
//!!!!------HAU ALDATZEKO-------!!!!------------------------(pentsatu nun jarri)------------------------------------------

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
