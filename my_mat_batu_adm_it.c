#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <empi.h>
#include <stdbool.h>


#define ROWS 111
#define COLS 64

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
            printf("%3d |", matrix[COLS*i+j]);
        }
        printf("\n");
    }
}

void sumMatrix(int matrix1[ROWS*COLS], int matrix2[ROWS*COLS], int resultMatrix[ROWS*COLS]){
    int i, j;

    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            resultMatrix[COLS*i+j] = matrix1[COLS*i+j] + matrix2[COLS*i+j];
        }
        printf("\n");
    }
}

void compareMatrix(int matrix1[ROWS*COLS], int matrix2[ROWS*COLS]) {
    int i, j;

    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            if (matrix1[COLS*i+j]==matrix2[COLS*i+j])
            {
                printf("true |");
            }else{
                printf("false |");
            }
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {

    int it=0;
    int itmax=ROWS*COLS;

    int *matrix1 = NULL;
    int *matrix2 = NULL;
    int *result = NULL;

    matrix1 = (int *)malloc(ROWS * COLS * sizeof(int));
    matrix2 = (int *)malloc(ROWS * COLS * sizeof(int));
    result = (int *)malloc(ROWS * COLS * sizeof(int));

    MPI_Datatype intercalate_type;                                  //A type to gather intercalated

    int world_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(ADM_COMM_WORLD, &world_rank);
    MPI_Comm_size(ADM_COMM_WORLD, &world_size);

    int dim = ROWS*COLS;

    if (world_rank == 0) {
        srand(time(NULL));  // Initiate random seed
        generateRandomMatrix(matrix1);
        generateRandomMatrix(matrix2);
    }

    MPI_Bcast(matrix1, dim, MPI_INT, 0, ADM_COMM_WORLD);    //Broadcast both matrix
    MPI_Bcast(matrix2, dim, MPI_INT, 0, ADM_COMM_WORLD);

    // get process type
    int proctype;
    ADM_GetSysAttributesInt ("ADM_GLOBAL_PROCESS_TYPE", &proctype);

    // if process is native
    if (proctype == ADM_NATIVE) {
        
        printf ("Rank(%d/%d): Process native\n", world_rank, world_size);
        
    // if process is spawned
    } else {
        printf ("Rank(%d/%d): Process spawned\n", world_rank, world_size);
    }
    
    /* set max number of iterations */
    ADM_RegisterSysAttributesInt ("ADM_GLOBAL_MAX_ITERATION", &itmax);

    /* get actual iteration for new added processes*/
    ADM_GetSysAttributesInt ("ADM_GLOBAL_ITERATION", &it);

    /* starting monitoring service */
    ADM_MonitoringService (ADM_SERVICE_START);

    // init last world size, flags, counters and cache variables
    int last_world_size = world_size;
    int last_it = it;
    bool send = false;
    bool sync = false;
    int saved_last_world_size = world_size;
    int ec = 0;

    // start loop
    for (; it < itmax; it++)        //All needed iterations to calculate the matrix sum
    {
        //Select hints on specific iterations
        int procs_hint = 0;
        int excl_nodes_hint = 0;

        if ( (it == 2*(itmax/10)) || (it == 4*(itmax/10)) ){
            procs_hint = 2;
            excl_nodes_hint = 0;
            sync = true;
            ADM_RegisterSysAttributesInt ("ADM_GLOBAL_HINT_NUM_PROCESS", &procs_hint);
            ADM_RegisterSysAttributesInt ("ADM_GLOBAL_HINT_EXCL_NODES", &excl_nodes_hint);
            //printf("Rank (%d/%d): Iteration:= %d, procs_hint=%d, excl_nodes_hint=%d\n", world_rank, world_size, it, procs_hint, excl_nodes_hint);
        } else if ( (it == 6*(itmax/10)) || (it == 8*(itmax/10)) ){
            procs_hint = -2;
            excl_nodes_hint = 0;
            send = true;
            ADM_RegisterSysAttributesInt ("ADM_GLOBAL_HINT_NUM_PROCESS", &procs_hint);
            ADM_RegisterSysAttributesInt ("ADM_GLOBAL_HINT_EXCL_NODES", &excl_nodes_hint);
            //printf("Rank (%d/%d): Iteration:= %d, procs_hint=%d, excl_nodes_hint=%d\n", world_rank, world_size, it, procs_hint, excl_nodes_hint);
        }else if (it == (itmax-1)){
            send = true;
        }
        
        // update message if new spawned processes
        if (last_world_size != world_size) {

            if (last_world_size<world_size)
            {
                MPI_Bcast(matrix1, dim, MPI_INT, 0, ADM_COMM_WORLD);    //Broadcast both matrix to spawned processes
                MPI_Bcast(matrix2, dim, MPI_INT, 0, ADM_COMM_WORLD);
            }
            saved_last_world_size = last_world_size;
            last_world_size = world_size;
            if (sync == true)
            {
                send = true;
            }
            
        }

        if (send == true)
        {
            if (sync == false)
            {
                MPI_Type_vector(ec, 1, last_world_size, MPI_INT, &intercalate_type);
                MPI_Type_commit(&intercalate_type);
                if (world_rank!=0)
                {
                    MPI_Send(&result[last_it+world_rank], 1, intercalate_type, 0, 0, ADM_COMM_WORLD);
                    ec = 0;
                }
                if (world_rank==0)
                {
                    for (int i = 1; i < last_world_size; i++)
                    {
                        MPI_Recv(&result[last_it+i], 1, intercalate_type, i, 0, ADM_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }else{
                MPI_Type_vector(ec, 1, saved_last_world_size, MPI_INT, &intercalate_type);
                MPI_Type_commit(&intercalate_type);
                if (world_rank!=0)
                {
                    MPI_Send(&result[last_it+world_rank], 1, intercalate_type, 0, 0, ADM_COMM_WORLD);
                    ec = 0;
                }
                if (world_rank==0)
                {
                    for (int i = 1; i < saved_last_world_size; i++)
                    {
                        MPI_Recv(&result[last_it+i], 1, intercalate_type, i, 0, ADM_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
                sync = false;
            }
            last_it = it;
            send = false;
        }
        
        
        /* start malelability region */
        ADM_MalleableRegion (ADM_SERVICE_START);


        if ((it-last_it)%world_size == world_rank)        //Divide workload in processes
        {
            result[it] = matrix1[it] + matrix2[it];
            ec++;
            printf("Rank (%d/%d): Iteration:= %d, %d + %d = %d\n", world_rank, world_size, it, matrix1[it], matrix2[it], result[it]);
        }else{
            //printf("Rank (%d/%d): Iteration= %d\n", world_rank, world_size, it);
        }

        //sleep(1);

        

        // update the iteration value
        ADM_RegisterSysAttributesInt ("ADM_GLOBAL_ITERATION", &it);        
        
        // ending malleable region
        int status;
        status = ADM_MalleableRegion (ADM_SERVICE_STOP);

        
        if (status == ADM_ACTIVE) {
            // updata rank and size
            MPI_Comm_rank(ADM_COMM_WORLD, &world_rank);
            MPI_Comm_size(ADM_COMM_WORLD, &world_size);
        } else {
            // end the process
            break;
        }
    }

    /* ending monitoring service */
    ADM_MonitoringService (ADM_SERVICE_STOP);

    printf("Rank (%d/%d): End of loop \n", world_rank, world_size);

    MPI_Finalize();

    if (world_rank == 0) {
        int resultMatrix[ROWS*COLS];

        printf("Matrix 1:\n");
        printMatrix(matrix1);

        printf("\nMatrix 2:\n");
        printMatrix(matrix2);

        sumMatrix(matrix1, matrix2, resultMatrix);

        printf("\nSum result (non parallel):\n");
        printMatrix(resultMatrix);

        printf("\nSum result:\n");
        printMatrix(result);

        printf("\nCompare:\n");
        compareMatrix(result, resultMatrix);

        free(matrix1);
        free(matrix2);
        free(result);
    }

    
    return 0;
}
