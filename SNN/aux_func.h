#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <random>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#define PI (3.141592653589793)
#define OOEPS1 (40)
#define OOEPS2 (3600)
#define Q (.002)
#define F (1.16)
#define PHI (.04)
#define U0 (0.661614)
#define V0 (0.198013)
#define W0 (0.996766)
#define DT (0.0007)
#define ALPHA (65.933)
#define BETA (2/.409086)
#define MAX (4)

using namespace Eigen;

typedef struct node Node;
typedef struct connection Connection;
struct node{
	int activate;
	double fire_time;
	double u;
	double w;
	double v;
	double *deriv_fire;
	double *conc_fire;
};
double dudt(double u,double v, double w);
double dvdt(double u,double v, double w);
double dwdt(double u,double v, double w);
double feed_one(Node **network, double **weights_1, double **weights_2,
 MatrixXd x, int inp_node, int hidden_node, int output_node, MatrixXd y, int max);
void reset_network(Node **network, int inp, int hidden, int output, int max);
double *back_prop_1(Node **network, double **weights_2, MatrixXd y, 
	int hidden_node, int output_node);
double *back_prop_2(Node **network, double **weights_1, double **weights_2, double *del,
	int input_node, int hidden_node, int output_node);
void print_network(Node **network, int inp_node, int hidden_node, int output_node);
double add_it(Node *node, int j, int i, double **weights);
void update(Node **network, int inp_node, int hidden_node, int output_node);
void update_node(Node *node);
MatrixXd get_data();
MatrixXd get_y();
MatrixXd weights1();
MatrixXd weights2();
void print_weights(double **weights, int, int);