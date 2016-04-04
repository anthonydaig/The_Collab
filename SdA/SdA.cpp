#include <math.h>
#include "Eigen/Dense"
#include <random>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

using namespace Eigen;


class SdA {


private:
	void initializeTheta(int inp, int out, MatrixXd *theta);
	void initialize(int *num_nodes, int xcols, int ycols, int nodes);
	void add_bias(MatrixXd *x);
	void sigmoid(MatrixXd x, MatrixXd *x1);
	void sigmoid_ele(double *y);
	void mult_deriv(MatrixXd, MatrixXd, MatrixXd *x);

public:
	int inp;
	int hidden;
	double mu = 0;
	double sd = 1.5;
	MatrixXd *weights;
	MatrixXd ones;
	MatrixXd encoded;
	MatrixXd decoded;
	MatrixXd weight_recon;


	SdA() { }
	void learn(MatrixXd, int*, int, int);
	MatrixXd corrupt_input(MatrixXd);
	void hidden_val(MatrixXd, MatrixXd);
	void reconstruct_init();
	void set_noise_mean(double mu1){mu = mu1;}
	void set_noise_sd(double sd1){sd= sd1;}
	void dA(MatrixXd, int, int);

};

void SdA::sigmoid(MatrixXd x, MatrixXd *x1)
{
	*x1 = (1./(1. + exp(-1.*x.array()))).matrix();	
}

void SdA::sigmoid_ele(double y)
{
	return (1./(1. + exp(-1.*(y))));
}


void SdA::initializeTheta(int inp, int out, MatrixXd *theta)
{
	double eps = sqrt(6)/sqrt((double)inp + (double)out);
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(-eps,eps);
	(*theta).conservativeResize(inp, out);
	for(int i = 0; i < inp; ++i)
	{
		for(int j = 0; j < out; ++j)
		{
			(*theta)(i,j) = distribution(generator);
		}
	}
}


//corrupt_input
MatrixXd SdA::corrupt_input(MatrixXd X)
{


	MatrixXd corrupt(X.rows(), X.cols());
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(mu,sd); //make this binom



	for (int i = 0; i < X.rows(); ++i)
	{
		for (int j = 0; j < X.cols(); ++j)
		{
			corrupt(i,j) = X(i,j) + distribution(generator);
		}
	}
	return corrupt;
}



void SdA::add_bias(MatrixXd *x)
{
	x->conservativeResize(x->rows(), x->cols()+1);
    x->col(x->cols()-1).setOnes();
}


void SdA::initialize(int *num_nodes, int xcols, int ycols, int nodes)
{
		initializeTheta(num_nodes[0], xcols+1, &(weights[0]));

		for (int i = 1; i < nodes; ++i)
		{
			initializeTheta(num_nodes[i], num_nodes[i-1] + 1, &(weights[i]));
		}

		initializeTheta(ycols, num_nodes[nodes-1] + 1, &(weights[nodes]));

}


void SdA::hidden_val(MatrixXd x, MatrixXd weight)
{
	add_bias(&x);
	encoded = x*weight.transpose();
	sigmoid(encoded, &encoded);
}

void SdA::reconstruct_init()
{
	add_bias(&encoded);
	decoded = encoded*weight_recon.transpose();
}

void SdA::mult_deriv(MatrixXd X, MatrixXd *y)
{

	for (int i = 0; i < X.rows(); ++i)
	{
		for (int j = 0; j < X.cols(); ++j)
		{
			(*y)(i,j) = (X(i,j) - decoded(i,j))


			// (1 - sigmoid_ele(decoded(i,j)))
			//*sigmoid_ele(decoded(i,j));
		}
	}
}

void SdA::dA(MatrixXd X, int out_nodes, int layer_num)
{
	initializeTheta(X.cols(), out_nodes+1, &weight_recon);
	hidden_val(X, weights[layer_num]);
	reconstruct_init();


	MatrixXd diff;

	ones = MatrixXd::Ones(X.rows(), X.cols());

	int max_iter = 2;

	int i = 0;

	while(i < max_iter){

		mult_deriv(X, &diff);

















		i++;



	}








}



void SdA::learn(MatrixXd X, int *nodes, int layers, int outputs)
{

	weights = new MatrixXd[layers+1];

	// initializeTheta(4, 10,&weight_recon);

	std::cout << X.cols() << std::endl;

	initialize(nodes, X.cols(), outputs, layers);

	dA(X, nodes[0], 0);

	// hidden_val(X, weights[0]);

	// reconstruct_init();




	std::cout << "encoded:\n";
	std::cout << encoded << std::endl;
	std::cout << "decoded:\n";
	std::cout << decoded << std::endl;


}






int main(int argv, char** argc){

	MatrixXd x(6,4);
	x <<	0, 0, 3, 6,
			1, 1, 3, 4,
			1, 0, 1, 2,
			0, 1, 1, 1,
			4, 1, 0, 9,
			4, 8, 6, 5;

	SdA sda;

	std::cout << x << std::endl;


	int nodes[3];
	nodes[0] = 9;
	nodes[1] = 6;
	nodes[2] = 9;


	std::cout << "\n\n"; 

	sda.learn(x, nodes, 3, 2);




	



	return 0;
}