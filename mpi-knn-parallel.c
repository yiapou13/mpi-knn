#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include "mat.h"

#define NN 30 //number of neigbours
#define max 10 //number of classes

struct timeval startwtime, endwtime;
double seq_time;

typedef struct
{
	double distance;
	int idx;
	int label;
} neighbour;

void omp_set_num_threads(int);
int omp_get_num_threads();

int compare(const void *s1, const void *s2)
{
	double a = ((neighbour*)s1)->distance;
	double b = ((neighbour*)s2)->distance;
	if (a < b) return -1;
	else if (a == b) return 0;
	else return 1;
}

void clear(int *table, int size)
{
	for (int i = 0; i < size; i++)
		table[i] = 0;
}

double **alloc_2d_double(int rows, int cols)
{
    double *data = (double *)malloc(rows*cols*sizeof(double));
    double **array= (double **)malloc(rows*sizeof(double*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}

int main(int argc, char **argv)
{	
	int numprocs, id; //MPI variables
	int procs = atoi(argv[1]); //MPI processes
	int threads = atoi(argv[2]); //OpenMP threads
	size_t n, m;
	int i, j, k, matches = 0;
	
	MPI_Init ( &argc, &argv );
	MPI_Status status;
	MPI_Comm_size ( MPI_COMM_WORLD, &numprocs );
	MPI_Comm_rank ( MPI_COMM_WORLD, &id );
	
	//Accesing MATLAB .mat file 
	MATFile *pmat;
	pmat = matOpen("mnist_train.mat", "r");
	mxArray *Xmat, *Lmat;
	Xmat = matGetVariable(pmat, "train_X");
	Lmat = matGetVariable(pmat, "train_labels");
	
	m = mxGetM(Xmat); //number of rows
	n = mxGetN(Xmat); //number of columns
	
	int *labels = (int *)malloc((m / numprocs) * sizeof(int));
	int *class = (int *)malloc(max * sizeof(int));

	double *orig_point = (double*) mxGetPr(Xmat); // first ellement of array
	double *label_point = (double*) mxGetPr(Lmat);
	
	double **matrix = alloc_2d_double(m / procs, n + 2);
	double **matrix_temp = alloc_2d_double(m / procs, n + 2);
	
	neighbour **kNNeighbours = (neighbour **)malloc(m / procs * sizeof(neighbour *));
	for (i = 0; i < m / procs; i++)
		kNNeighbours[i] = (neighbour *)malloc(n * sizeof(neighbour));
	
	//Initializing distance matrix
	for (k = 0; k < m / procs; k++)
	{
		for (i = 0; i < NN; i++)
		{
			kNNeighbours[k][i].distance = INFINITY;
		}
	}
	
	//Fillin coordinate matrix for each point.
	//1st extra cell: point ID
	//2nd extra cell: point label
	for (int k = 0; k < m / procs; k++)
	{
		for (int i = 0; i < n; i++)
		{	
			matrix[k][i] = orig_point[k + (id * m / procs) + (i * m)];
		}
		
		matrix[k][n] = k + (id * m / procs) + 1;
		matrix[k][n + 1] = label_point[k + (id * m / procs)];
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	mxDestroyArray(Xmat);
	mxDestroyArray(Lmat);
	
	matClose(pmat);
		
	MPI_Barrier(MPI_COMM_WORLD);
	
	if (id == 0) gettimeofday( &startwtime, NULL );// start timer
	
/**** 1st communication and KNN calculation ****/
	//Sending starting matrix
	if (id == 0)
	{
		int dest = id + 1;
		int source = numprocs - 1;
		
		MPI_Recv ( &(matrix_temp[0][0]), m / procs * (n + 2), MPI_DOUBLE, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
		MPI_Send ( &(matrix[0][0]), m / procs * n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD );
		printf("DONE\n");
	}
	else if (id == numprocs - 1)
	{
		int dest = 0;
		int source = id - 1;

		MPI_Send ( &(matrix[0][0]), m / procs * n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD );
		MPI_Recv ( &(matrix_temp[0][0]), m / procs * n, MPI_DOUBLE, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
		printf("done\n");
	}
	else
	{
		int dest = id +1;
		int source = id - 1;
		
		MPI_Recv ( &(matrix_temp[0][0]), m / procs * n, MPI_DOUBLE, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Send ( &(matrix[0][0]), m / procs * n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD );
		printf("DOne\n");
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	//Initializing matrix for the rest of the process
	double **matrix_send = alloc_2d_double(m / procs , n + 2);
	
	omp_set_num_threads(threads);
	
	//KNN calculation with OpenMP
	#pragma omp parallel for
	for (k = 0; k < m / procs; k++)
	{
		for (int i = 0; i < m / procs; i++)
		{
			double S = 0;

				for (j = 0; j < n; j++)
				{
					double Da, Db;

					Da = matrix[k][j];
					Db = matrix[i][j];
					S = S + pow((Da - Db), 2);
					matrix_send[i][j] = matrix_temp[i][j]; //copying received matrix
				}
			
				if ((sqrt(S) < kNNeighbours[k][NN-1].distance) && (sqrt(S) != 0))
				{
					kNNeighbours[k][NN-1].distance = sqrt(S);
					kNNeighbours[k][NN-1].idx = matrix[i][n];
					kNNeighbours[k][NN-1].label = matrix[i][n + 1];
					qsort(kNNeighbours[k], NN, sizeof(neighbour), compare);
				}
		}

	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
/**** N-th communication and KNN calculation ****/
	 
	for (int p = 0; p < procs - 1; p++)
	{
		//Sending previously received maxtrix
		if (id == 0)
		{
			int dest = id + 1;
			int source = numprocs - 1;

			MPI_Recv ( &(matrix_temp[0][0]), m / procs * (n + 2), MPI_DOUBLE, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
			MPI_Send ( &(matrix_send[0][0]), m / procs * (n + 2), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD );
			printf("DONE\n");
		}
		else if (id == numprocs - 1)
		{
			int dest = 0;
			int source = id - 1;

			MPI_Send ( &(matrix_send[0][0]), m / procs * (n + 2), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD );
			MPI_Recv ( &(matrix_temp[0][0]), m / procs * (n + 2), MPI_DOUBLE, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
			printf("done\n");
		}
		else
		{
			int dest = id +1;
			int source = id - 1;

			MPI_Recv ( &(matrix_temp[0][0]), m / procs * (n + 2), MPI_DOUBLE, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			MPI_Send ( &(matrix_send[0][0]), m / procs * (n + 2), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD );\
			printf("DOne\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
		
		//KNN calculation with OpenMP
		#pragma omp parallel for
		for (k = 0; k < m / procs; k++)
		{
			for (int i = 0; i < m / procs; i++)
			{
				double S = 0;

				for (j = 0; j < n; j++)
				{
					double Da, Db;

					Da = matrix[k][j];
					Db = matrix_temp[i][j];
					S = S + pow((Da - Db), 2);
					matrix_send[i][j] = matrix_temp[i][j]; //copying received matrix
				}
				
				if ((sqrt(S) < kNNeighbours[k][NN-1].distance) && (sqrt(S) != 0))
				{
					kNNeighbours[k][NN-1].distance = sqrt(S);
					kNNeighbours[k][NN-1].idx = matrix_temp[i][n];
					kNNeighbours[k][NN-1].label = matrix_temp[i][n + 1];
					qsort(kNNeighbours[k], NN, sizeof(neighbour), compare);
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);	
	}

	MPI_Barrier(MPI_COMM_WORLD);
		
	if (id == 0) gettimeofday( &endwtime, NULL );// end timer

	if (id == 0) seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
	
	MPI_Barrier(MPI_COMM_WORLD);
	
/**** Matching labels ****/
	printf("Process :%d before\n", id);
	
	for (k = 0; k < m / procs; k++)
	{
		int most = 0;
		clear(class, max);
		
		for (i = 0; i < NN; i++)
		{
			class[kNNeighbours[k][i].label - 1] = class[kNNeighbours[k][i].label - 1] + 1;
		}
		
		for (j = 0; j < max; j++)
		{
			if (class[j] > most || ((class[j] == most) && ((j+1) == (kNNeighbours[k][0].label - 1) ))) most = j + 1;
		}
		
		labels[k] = most;
		if (labels[k] == matrix[k][785]) matches++;
	}
	
	printf("Match percentage: %d\n", matches);
	if (id == 0) printf("KNN time: %f", seq_time);
	
	MPI_Finalize ( );
	
	return 0;
}



