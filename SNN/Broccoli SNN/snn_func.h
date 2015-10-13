#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <random>
#include <Eigen/Dense>

typedef struct node Node;
struct node{
	int activate;
	double fire_time;
	double *deriv_fire;
	double *conc_fire;
};



void reset_network(Node **network, int *a, const int);

void print_network(Node **network, int *a, const int);




