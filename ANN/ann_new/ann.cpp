#include <math.h>
#include "Eigen/Dense"
#include <random>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#define gucci (1)
#define MAXX (0)

using namespace Eigen;

class ann {

private:
	MatrixXd train(MatrixXd, MatrixXd, bool, int*, MatrixXd*);

public:
	const double eta = 0.1;
	const int iter = 1000;

	ann() { }
	MatrixXd normalize(MatrixXd);
	MatrixXd sigmoid(MatrixXd);
	MatrixXd sigmoidDeriv(MatrixXd);
	MatrixXd initializeTheta(int, int);
	MatrixXd* gradientStep(MatrixXd, MatrixXd, int, bool, std::vector<int>);
	MatrixXd* nncost(MatrixXd, MatrixXd, int, MatrixXd*, int, double*, bool, int);
};

MatrixXd ann::normalize(MatrixXd x)
{
	double mean, sd;
	for(int i = 0; i < x.cols() - 1; ++i)
	{
		mean = x.col(i).mean();
		
		sd = sqrt((((x.col(i).array() - mean)/(x.rows()-1)).matrix().transpose())*(((x.col(i).array() - mean)/(x.rows()-1)).matrix()));

		x.col(i) = ((x.col(i).array() - mean)/sd).matrix();

	}
	return x;
}

MatrixXd ann::sigmoid(MatrixXd x)
{
	return (1./(1. + exp(-1.*x.array()))).matrix();	

}

MatrixXd ann::sigmoidDeriv(MatrixXd x)
{
	return (sigmoid(x).array()*(1-sigmoid(x).array())).matrix();
}

MatrixXd ann::initializeTheta(int inp, int out)
{
	double eps = sqrt(6)/sqrt((double)inp + (double)out);
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(-eps,eps);
	MatrixXd theta(inp, out);
	for(int i = 0; i < inp; ++i)
	{
		for(int j = 0; j < out; ++j)
		{
			theta(i,j) = distribution(generator);
			distribution.reset();
		}
	}
	return theta;
}

// MatrixXd ann::train(MatrixXd x, MatrixXd y, bool normalize_flag, int* cost_point, MatrixXd* thetas)
// {
// 	double lambda = 5;


// 	if(normalize_flag)
// 	{
// 		x = normalize(x);
// 	}

// 	double cost = 0;
	

// 	MatrixXd a_i(x.cols(), 1);

// 	for(int i = 0; i < x.rows(); ++i)
// 	{
// 		a_i = x.row(i);
// 	}

// 	*cost_point = cost;
// 	return *thetas;


MatrixXd* ann::gradientStep(MatrixXd x, MatrixXd y, int iter, bool bias, std::vector<int> nodes)
{
	// nodes is a list of the number of nodes we want in a hidden layer

	double lambda = 5;
	double eta = .1;
	double cost_point[iter];


	// Add a bias column to the data if not added
	if(!bias)
	{
		x.conservativeResize(x.rows(), x.cols()+1);
		x.col(x.cols()-1).setOnes();
	}



	int inps = x.cols(); // Number of input nodes
	int outp = y.cols(); // Number of final output nodes

	int hidden = nodes.size(); // Number of hidden layers

	MatrixXd* thetas = new MatrixXd[hidden+1]; // hidden + 2 layers, hidden + 1 connections
	MatrixXd* deltas = new MatrixXd[hidden+1];

	thetas[0] = initializeTheta(nodes.at(0), inps);

	std::cout << hidden << std::endl;

	for(int i = 0; i < hidden - 1; ++i)
	{
		thetas[i+1] = initializeTheta(nodes.at(i+1), nodes.at(i) + 1);
	}

	thetas[hidden] = initializeTheta(outp, nodes.at(hidden-1) + 1);


	for(int i = 0; i < hidden+1; ++i)
	{
		std::cout << thetas[i] << std::endl;
		std::cout << std::endl;
	}

	int i = 0;
	while(i < iter)
	{
		deltas = nncost(x, y, ++i, thetas, iter, cost_point, 1, hidden+1);
		for(int j = 0; j < hidden + 2; ++j){
			thetas[j] = thetas[j] - eta*deltas[j];
		}

		// Length of del's is going to be length of thetas
		// for(int j = 0; j < )
	}

	return thetas;
}

MatrixXd* ann::nncost(MatrixXd x, MatrixXd y, int count, MatrixXd* thetas, int iter, double* cost, bool normalize_flag, int theta_len)
{
	double lambda = 5;

	std::cout << theta_len << std::endl;

	if(normalize_flag)
	{
		x = normalize(x); // this normalize function assumes the last column is a bias column and thus does not normalize that column
	}

	// Bias term handled in ann::gradientStep

	double this_cost = 0;

	MatrixXd a_i(x.cols(), 1);
	MatrixXd* a_s = new MatrixXd[theta_len+1]; // keep tabs on this initial size
	MatrixXd* z_s = new MatrixXd[theta_len];
	MatrixXd* dels = new MatrixXd[theta_len];
	MatrixXd* deltas = new MatrixXd[theta_len];
	MatrixXd* final_delta = new MatrixXd[theta_len];
	MatrixXd z2, a2, h_theta, y_temp, z_i, del_prev, del_i;
	
	int h_theta_len;

	double log_1, log_2, term_1, term_2;

	for(int i = 0; i < x.rows(); ++i)
	{
		a_s[0] = x.row(i);

		std::cout << a_s[0].cols() << std::endl << thetas[0].transpose().rows() << std::endl;

		for(int j = 0; j < theta_len; ++j)
		{

			std::cout << a_s[j]*thetas[j].transpose() << std::endl;

			z_s[j] = a_s[j]*thetas[j].transpose();

			std::cout << sigmoid(z_s[j]) << std::endl;

			a_s[j+1] = sigmoid(z_s[j]);

			if(j != theta_len - 1)
			{	
				a_s[j+1].conservativeResize(a_s[j+1].rows(), a_s[j+1].cols()+1);
				a_s[j+1](a_s[j+1].cols()-1) = 1;
			}


			// z_s[j] = z2;
			// a2 = sigmoid(z2);
			// a_s[j+1] << 1, a2;

		}

		std::cout << "here me" << std::endl;

		h_theta = a_s[theta_len];

		std::cout << h_theta << std::endl;


		h_theta_len = thetas[theta_len - 1].rows();

		std::cout << h_theta_len << std::endl;
		
		y_temp = y.row(i);

		std::cout << y_temp << std::endl;

		if(count == iter - 1)
		{
			std::cout << h_theta << std::endl;
			std::cout << y_temp << std::endl;
		}

		for(int j = 0; j < h_theta_len; ++j)
		{
			log_1 = log(h_theta(j));
			log_2 = log(1-h_theta(j));
			term_1 = -log_1*y_temp(j);
			term_2 = log_2*(1-y_temp(j));
			this_cost += term_1 + term_2;

			std::cout << this_cost << std::endl;
		}

		// the dels[theta_len - 1] entry
		dels[theta_len - 1] = a_s[theta_len] - y.row(i);

		std::cout << "dels[theta_len-1]:\n" << dels[theta_len - 1] << std::endl;

		for(int j = theta_len - 1; j > 0; --j)
		{
			// .row(0) may not be general


			std::cout << "thetas[j]:\n" << thetas[j] << std::endl;
			std::cout << "dels[j]:\n" << dels[j] << std::endl;
			std::cout << "mult:\n" << dels[j]*thetas[j] << std::endl;
			std::cout << "sigmoidDeriv:\n" << sigmoidDeriv(z_s[j-1]) << std::endl;

			
			dels[j-1] = (dels[j]*thetas[j])*sigmoidDeriv(z_s[j-1]);
			std::cout << dels[j-1] << std::endl;
		}

		for(int j = 0; j < theta_len; ++j)
		{
			deltas[j] = dels[j].transpose()*a_s[j]/theta_len;
		}

		if(i == 0)
		{
			final_delta = deltas;
		} else {
			for(int k = 0; k < theta_len; ++k)
			{
				final_delta[k] += deltas[k];
			}
		}

	}

	cost[count - 1] = this_cost;

	return final_delta;
}


int main(int argv, char** argc){
	ann net;
	MatrixXd x(4,4);
	MatrixXd y(4,2);
	x <<	0, 0, 3, 6,
			1, 1, 3, 4,
			1, 0, 1, 2,
			0, 1, 1, 1;
	y <<	0, 1,
			0, 1,
			1, 0,
			1, 0;
	std::cout << x << std::endl;
	std::cout << y << std::endl;

	std::vector<int> nodes;
	nodes.push_back(3);
	nodes.push_back(2);
	
	net.gradientStep(x, y, 100, 0, nodes);


	return 0;
}
