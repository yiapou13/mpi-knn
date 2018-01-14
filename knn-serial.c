#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
#include "mat.h"

#define NN 30 // number of neighbours
#define max 10 //number of labels (classes)

struct timeval startwtime, endwtime;
double seq_time;

typedef struct
{
	double distance;
	int idx;
} neighbour;

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

int main(int argc, char *argv[])
{		
	MATFile *pmat;

	pmat = matOpen("mnist_train.mat", "r");
	
	mxArray *Xmat, *Lmat;
	
	size_t n, m;
	int i, j, k, matches = 0;
	
	Xmat = matGetVariable(pmat, "train_X");
	
	m = mxGetM(Xmat); // number of rows
	n = mxGetN(Xmat); // number of columns
	
	Lmat = matGetVariable(pmat, "train_labels");

	int class[max], labels[m];
	neighbour kNeighbours[m][NN];
		
	for (k = 0; k < m; k++)
	{
		for (i = 0; i < NN; i++)
		{
			kNeighbours[k][i].distance = INFINITY;
		}
	}
	
	printf("Number of Classes: %d\n", max);
	
	double *orig_point = (double*) mxGetPr(Xmat); // first ellement of array
	double *comp_point = (double*) mxGetPr(Xmat); // first ellement of array
	
	gettimeofday( &startwtime, NULL );// start timer
	
	for (k = 0; k < m; k++)
	{
		for (int i = 0; i < m; i++)
		{
			double S = 0;
				
			for (j = 0; j < n; j++)
			{
				double Da, Db;

				Da = orig_point[k + (j * m)];
				Db = comp_point[i + (j * m)];
				S = S + pow((Da - Db), 2);
			}
			if ((sqrt(S) < kNeighbours[k][NN-1].distance) && (sqrt(S) != 0))
			{
				kNeighbours[k][NN-1].distance = sqrt(S);
				kNeighbours[k][NN-1].idx = i + 1;
				qsort(kNeighbours[k], NN, sizeof(neighbour), compare);
			}
		}
	}
	gettimeofday( &endwtime, NULL );// end timer
	
	seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
	
	printf("Sorting done\nClock time = %f\n", seq_time);
	
	matClose(pmat);
	
	mxDestroyArray(Xmat);
	
	//Matching labels
	
	pmat = matOpen("mnist_train.mat", "r");
	
	Lmat = matGetVariable(pmat, "train_labels");
	double *start_point = (double*) mxGetPr(Lmat); // first ellement of array
	
	for (k = 0; k < m; k++)
	{
		int most = 0;
		clear(class, max);
		
		for (i = 0; i < NN; i++)
		{
			class[((int)start_point[kNeighbours[k][i].idx - 1])- 1] = class[((int)start_point[kNeighbours[k][i].idx - 1])- 1] + 1;
		}
		
		for (j = 0; j < max; j++)
		{
			if (class[j] > most || ((class[j] == most) && ((j+1) == (int)start_point[kNeighbours[k][0].idx - 1] ))) most = j + 1;
		}
		
		labels[k] = most;
		if (labels[k] == start_point[k]) matches++;
	}
	
	printf("Matches: %d\n", matches);
	
	return 0;
}



