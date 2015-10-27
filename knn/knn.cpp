#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double dist(int m, double pointa[m], double pointb[m]);
void all_dist(int m, int n, double data[m][n], double alldist[n][n], int ref[n][n]);
void knn(int k, int n, int ref[n][n], int kref[n][k]);

//Matrix Operations
void printMatrix(int m, int n, double matrix[m][n]);
void printMatrixi(int m, int n, int matrix[m][n]);
void transpose(int m, int n, double matrix[n][n], double trans[n][n]);
void printArray(int n, double arrray[n]);
void printArrayi(int n, int arrray[n]);

//QuickSort stuff
void quickSort(double a[], int l, int h);
void quickSortRef(double a[], int ref[], int l, int h);
void swap(double* a, double* b);
void swapInt();
int partition(double arr[], int l, int h);
int partitionRef(double arr[], int ref[], int l, int h);

int main(int argc, char *argv[]) {

	int k = 4;
	k = k + 1;

	//number of points
	int n = 5;
	int nIRIS = 150;

	//number of characteristics per point
	int m = 4;

	double point1[4] = { .1,.2,.3,.4 };
	double point2[4] = { .4,.3,.2,.1 };
	double point3[4] = { .1,.1,.1,.1 };
	double point4[4] = { .2,.2,.2,.2 };
	double point5[4] = { .4,.5,.6,.7 };

	double points[5][4]  = {
		{ .1,.2,.3,.4 },
		{ .4,.3,.2,.1 },
		{ .1,.1,.1,.1 },
		{ .2,.2,.2,.2 },
		{ .4,.5,.6,.7 },	};

	double iris[150][4] = {
		{5.1, 3.5, 1.4, 0.2},
		{4.9, 3.0, 1.4, 0.2},
		{4.7, 3.2, 1.3, 0.2},
		{4.6, 3.1, 1.5, 0.2},
		{5.0, 3.6, 1.4, 0.2},
		{5.4, 3.9, 1.7, 0.4},
		{4.6, 3.4, 1.4, 0.3},
		{5.0, 3.4, 1.5, 0.2},
		{4.4, 2.9, 1.4, 0.2},
		{4.9, 3.1, 1.5, 0.1},
		{5.4, 3.7, 1.5, 0.2},
		{4.8, 3.4, 1.6, 0.2},
		{4.8, 3.0, 1.4, 0.1},
		{4.3, 3.0, 1.1, 0.1},
		{5.8, 4.0, 1.2, 0.2},
		{5.7, 4.4, 1.5, 0.4},
		{5.4, 3.9, 1.3, 0.4},
		{5.1, 3.5, 1.4, 0.3},
		{5.7, 3.8, 1.7, 0.3},
		{5.1, 3.8, 1.5, 0.3},
		{5.4, 3.4, 1.7, 0.2},
		{5.1, 3.7, 1.5, 0.4},
		{4.6, 3.6, 1.0, 0.2},
		{5.1, 3.3, 1.7, 0.5},
		{4.8, 3.4, 1.9, 0.2},
		{5.0, 3.0, 1.6, 0.2},
		{5.0, 3.4, 1.6, 0.4},
		{5.2, 3.5, 1.5, 0.2},
		{5.2, 3.4, 1.4, 0.2},
		{4.7, 3.2, 1.6, 0.2},
		{4.8, 3.1, 1.6, 0.2},
		{5.4, 3.4, 1.5, 0.4},
		{5.2, 4.1, 1.5, 0.1},
		{5.5, 4.2, 1.4, 0.2},
		{4.9, 3.1, 1.5, 0.1},
		{5.0, 3.2, 1.2, 0.2},
		{5.5, 3.5, 1.3, 0.2},
		{4.9, 3.1, 1.5, 0.1},
		{4.4, 3.0, 1.3, 0.2},
		{5.1, 3.4, 1.5, 0.2},
		{5.0, 3.5, 1.3, 0.3},
		{4.5, 2.3, 1.3, 0.3},
		{4.4, 3.2, 1.3, 0.2},
		{5.0, 3.5, 1.6, 0.6},
		{5.1, 3.8, 1.9, 0.4},
		{4.8, 3.0, 1.4, 0.3},
		{5.1, 3.8, 1.6, 0.2},
		{4.6, 3.2, 1.4, 0.2},
		{5.3, 3.7, 1.5, 0.2},
		{5.0, 3.3, 1.4, 0.2},
		{7.0, 3.2, 4.7, 1.4},
		{6.4, 3.2, 4.5, 1.5},
		{6.9, 3.1, 4.9, 1.5},
		{5.5, 2.3, 4.0, 1.3},
		{6.5, 2.8, 4.6, 1.5},
		{5.7, 2.8, 4.5, 1.3},
		{6.3, 3.3, 4.7, 1.6},
		{4.9, 2.4, 3.3, 1.0},
		{6.6, 2.9, 4.6, 1.3},
		{5.2, 2.7, 3.9, 1.4},
		{5.0, 2.0, 3.5, 1.0},
		{5.9, 3.0, 4.2, 1.5},
		{6.0, 2.2, 4.0, 1.0},
		{6.1, 2.9, 4.7, 1.4},
		{5.6, 2.9, 3.6, 1.3},
		{6.7, 3.1, 4.4, 1.4},
		{5.6, 3.0, 4.5, 1.5},
		{5.8, 2.7, 4.1, 1.0},
		{6.2, 2.2, 4.5, 1.5},
		{5.6, 2.5, 3.9, 1.1},
		{5.9, 3.2, 4.8, 1.8},
		{6.1, 2.8, 4.0, 1.3},
		{6.3, 2.5, 4.9, 1.5},
		{6.1, 2.8, 4.7, 1.2},
		{6.4, 2.9, 4.3, 1.3},
		{6.6, 3.0, 4.4, 1.4},
		{6.8, 2.8, 4.8, 1.4},
		{6.7, 3.0, 5.0, 1.7},
		{6.0, 2.9, 4.5, 1.5},
		{5.7, 2.6, 3.5, 1.0},
		{5.5, 2.4, 3.8, 1.1},
		{5.5, 2.4, 3.7, 1.0},
		{5.8, 2.7, 3.9, 1.2},
		{6.0, 2.7, 5.1, 1.6},
		{5.4, 3.0, 4.5, 1.5},
		{6.0, 3.4, 4.5, 1.6},
		{6.7, 3.1, 4.7, 1.5},
		{6.3, 2.3, 4.4, 1.3},
		{5.6, 3.0, 4.1, 1.3},
		{5.5, 2.5, 4.0, 1.3},
		{5.5, 2.6, 4.4, 1.2},
		{6.1, 3.0, 4.6, 1.4},
		{5.8, 2.6, 4.0, 1.2},
		{5.0, 2.3, 3.3, 1.0},
		{5.6, 2.7, 4.2, 1.3},
		{5.7, 3.0, 4.2, 1.2},
		{5.7, 2.9, 4.2, 1.3},
		{6.2, 2.9, 4.3, 1.3},
		{5.1, 2.5, 3.0, 1.1},
		{5.7, 2.8, 4.1, 1.3},
		{6.3, 3.3, 6.0, 2.5},
		{5.8, 2.7, 5.1, 1.9},
		{7.1, 3.0, 5.9, 2.1},
		{6.3, 2.9, 5.6, 1.8},
		{6.5, 3.0, 5.8, 2.2},
		{7.6, 3.0, 6.6, 2.1},
		{4.9, 2.5, 4.5, 1.7},
		{7.3, 2.9, 6.3, 1.8},
		{6.7, 2.5, 5.8, 1.8},
		{7.2, 3.6, 6.1, 2.5},
		{6.5, 3.2, 5.1, 2.0},
		{6.4, 2.7, 5.3, 1.9},
		{6.8, 3.0, 5.5, 2.1},
		{5.7, 2.5, 5.0, 2.0},
		{5.8, 2.8, 5.1, 2.4},
		{6.4, 3.2, 5.3, 2.3},
		{6.5, 3.0, 5.5, 1.8},
		{7.7, 3.8, 6.7, 2.2},
		{7.7, 2.6, 6.9, 2.3},
		{6.0, 2.2, 5.0, 1.5},
		{6.9, 3.2, 5.7, 2.3},
		{5.6, 2.8, 4.9, 2.0},
		{7.7, 2.8, 6.7, 2.0},
		{6.3, 2.7, 4.9, 1.8},
		{6.7, 3.3, 5.7, 2.1},
		{7.2, 3.2, 6.0, 1.8},
		{6.2, 2.8, 4.8, 1.8},
		{6.1, 3.0, 4.9, 1.8},
		{6.4, 2.8, 5.6, 2.1},
		{7.2, 3.0, 5.8, 1.6},
		{7.4, 2.8, 6.1, 1.9},
		{7.9, 3.8, 6.4, 2.0},
		{6.4, 2.8, 5.6, 2.2},
		{6.3, 2.8, 5.1, 1.5},
		{6.1, 2.6, 5.6, 1.4},
		{7.7, 3.0, 6.1, 2.3},
		{6.3, 3.4, 5.6, 2.4},
		{6.4, 3.1, 5.5, 1.8},
		{6.0, 3.0, 4.8, 1.8},
		{6.9, 3.1, 5.4, 2.1},
		{6.7, 3.1, 5.6, 2.4},
		{6.9, 3.1, 5.1, 2.3},
		{5.8, 2.7, 5.1, 1.9},
		{6.8, 3.2, 5.9, 2.3},
		{6.7, 3.3, 5.7, 2.5},
		{6.7, 3.0, 5.2, 2.3},
		{6.3, 2.5, 5.0, 1.9},
		{6.5, 3.0, 5.2, 2.0},
		{6.2, 3.4, 5.4, 2.3},
		{5.9, 3.0, 5.1, 1.8} };
	
	int classifications[150] = { 0 };
	int ii = 0;
	for (; ii < 50; ii++) { classifications[ii] = 1; }
	for (; ii < 100; ii++) { classifications[ii] = 2; }
	for (; ii < 150; ii++) { classifications[ii] = 3; }
	printArrayi(nIRIS, classifications);

	//double pointsT[4][5] = { 0.0 };
	double irisT[4][150] = { 0.0 };

	//transpose(m, n, points, pointsT);
	transpose(m, nIRIS, iris, irisT);

	//double alldist[5][5] = { 0.0 };
	double alldistIRIS[150][150] = { 0.0 };
	
	//int ref[5][5] = { 0 };
	int refIRIS[150][150] = { 0 };

	all_dist(m, nIRIS, irisT, alldistIRIS, refIRIS);
	//printMatrixi(n, n, ref);

	//int kref[5][4] = { 0 };
	int krefIRIS[150][4] = { 0 };

	//knn(k, n, ref, kref);
	knn(k, nIRIS, refIRIS, krefIRIS);
	printMatrixi(nIRIS, k, krefIRIS);



}
// Cuts the distance matrix to length k
void knn(int k, int n, int ref[n][n], int kref[n][k] ) {
	
	// Shows the k nearest neighbors of all points
	int i = 0;
	for (; i < n; i++) {
		int j = 0;
		for (; j < k; j++) {
			kref[i][j] = ref[i][j];
		}
	}

//
}


// Calculates the distances for all sets of points
void all_dist(int m, int n, double data[m][n], double alldist[n][n], int ref[n][n]) {
	//printMatrix(m, n, data);

	int i = 0;
	for (; i < n; i++) {
		int j = 0;
		for (; j < n; j++) {
			ref[i][j] = j;
		}
	}

	//printMatrixi(n, n, ref);

	//initialize the matrix that will hold the transpose of the data
	double dataT[n][m];
	int k = 0;
	for (; k < n; k++) {
		int l = 0;
		for (; l < m; l++) {
			dataT[k][l] = 0.0;
		}
	}
	//printMatrix(n, m, dataT);

	//transpose the data
	transpose(n, m, data, dataT);
	//printMatrix(n, m, dataT);

	//fill in the distances matrix with distance values
	//***** can be made faster by only calculating a triangular of the values
	int ii = 0;
	for (; ii < n; ii++) {
		int jj = 0;
		for (; jj < n; jj++) {
		//***** for (; jj < ii; jj++) {
			alldist[ii][jj] = dist(m,dataT[ii],dataT[jj]);
		}
	}
	//printMatrix(n, n, alldist);
	int kk = 0;
	for (; kk < n; kk++) {
		quickSortRef(alldist[kk], ref[kk], 0, n-1);
	}
}


// Define the distance/cost function here
double dist(int m, double pointa[m], double pointb[m]) {

	double distance = 0.0;

	double distance_sq = 0.0;
	int i = 0;
	for (; i < m; i++) {
		distance_sq = distance_sq + (pointa[i]-pointb[i])*(pointa[i]-pointb[i]);
	}

	distance = sqrt(distance_sq);

	return distance;
}

// Prints a matrix
void printMatrix(int m, int n, double matrix[m][n]) {
	int i = 0;
	int j = 0;
	for (; i < m; i++) {
		printf("|");
		j = 0;
		for (; j < n; j++) {
			if (matrix[i] >= 0) {
				printf("%f,", matrix[i][j]);
			}
			else {
				printf("%f,", matrix[i][j]);
			}
		}
		printf("|\n");
	}
	printf("\n");
}

// Prints a matrix of ints
void printMatrixi(int m, int n, int matrix[m][n]) {
	int i = 0;
	int j = 0;
	for (; i < m; i++) {
		printf("|");
		j = 0;
		for (; j < n; j++) {
			if (matrix[i] >= 0) {
				printf("%i,", matrix[i][j]);
			}
			else {
				printf("%i,", matrix[i][j]);
			}
		}
		printf("|\n");
	}
	printf("\n");
}

// Prints an array or integers
void printArrayi(int n, int arrray[n]) {
	
	printf("|");
	int i = 0;
	for (; i < n; i++) {
		printf("%i,", arrray[i]);
	}
	printf("|\n");
}

// Prints an array of doubles
void printArray(int n, double arrray[n]) {

	printf("|");
	int i = 0;
	for (; i < n; i++) {
		printf("%f,", arrray[i]);
	}
	printf("|\n");
}

// Transposes a matrix
// (Input: n rows, m columns, Output: m rows, n columns)
void transpose(int m, int n, double matrix[n][m], double trans[m][n]) {

	int i = 0;
	for (; i < m; i++) {
		int j = 0;
		for (; j < n; j++) {
			trans[i][j] = matrix[j][i];
		}
	}
}

// Quicksort that simultaneously swaps references
void quickSortRef(double A[], int ref[], int l, int h) {
	if (l < h)
	{
		int p = partitionRef(A, ref, l, h); /* Partitioning index */
		quickSortRef(A, ref, l, p - 1);
		quickSortRef(A, ref, p + 1, h);
	}
}

/* A[] --> Array to be sorted, l  --> Starting index, h  --> Ending index */
void quickSort(double A[], int l, int h)
{
	if (l < h)
	{
		int p = partition(A, l, h); /* Partitioning index */
		quickSort(A, l, p - 1);
		quickSort(A, p + 1, h);
	}
}


void swap(double* a, double* b)
{
	double t = *a;
	*a = *b;
	*b = t;
}

void swapInt(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}

int partition(double arr[], int l, int h)
{
	double x = arr[h];
	int i = (l - 1);

	int j = l;
	for (; j <= h - 1; j++)
	{
		if (arr[j] <= x)
		{
			i++;
			swap(&arr[i], &arr[j]);
		}
	}
	swap(&arr[i + 1], &arr[h]);
	return (i + 1);
}

int partitionRef(double arr[], int ref[], int l, int h)
{
	double x = arr[h];
	int i = (l - 1);

	int j = l;
	for (; j <= h - 1; j++)
	{
		if (arr[j] <= x)
		{
			i++;
			swap(&arr[i], &arr[j]);
			swapInt(&ref[i], &ref[j]);
		}
	}
	swap(&arr[i + 1], &arr[h]);
	swapInt(&ref[i + 1], &ref[h]);
	return (i + 1);
}