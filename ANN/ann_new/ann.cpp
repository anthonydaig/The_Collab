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
	
	MatrixXd normalize(MatrixXd);
	void sigmoid(MatrixXd, MatrixXd*);
	void sigmoidDeriv(MatrixXd,MatrixXd*,int);
	void initializeTheta(int, int, MatrixXd*);
	void add_bias(MatrixXd*);
	MatrixXd *dels;
	MatrixXd *z_s;
	MatrixXd *a_s;
	MatrixXd *ones;
	void find_grad(MatrixXd X, MatrixXd Y, int i, int m, int nodes);
	// MatrixXd* nncost(MatrixXd, MatrixXd, int, MatrixXd*, int, double*, bool, int);

public:
	double eta = 0.1;
	double lambda = 1;
	int iter = 1000;
	MatrixXd *hidden_layers;


	ann() { }
	void set_learning_param(double eta1){eta = eta1;}
	void set_regularizer(double regular){lambda = regular;}
	void set_iterations(int num){iter = num;}
	// MatrixXd* gradientStep(MatrixXd, MatrixXd, int, std::vector<int>, bool);
	void learn(MatrixXd, MatrixXd, int, int*, bool prelearn =1/*whether or not we want stacked autoencoders (default 1)*/);

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

void ann::sigmoid(MatrixXd x, MatrixXd *x1)
{
	*x1 = (1./(1. + exp(-1.*x.array()))).matrix();	
}

void ann::sigmoidDeriv(MatrixXd x, MatrixXd *x1, int m)
{
	sigmoid(x,x1);
	(*x1).array();

	*x1 = (((ones[m]-(*x1)).array())*(x1->array())).matrix();

	// *x1 = ((*x1) * (ArrayXXf::setOnes(x.rows(), x.cols())-(*x1))).matrix();
	// return *x1 = (sigmoid(x, x1).array()*(1-sigmoid(x, x1).array())).matrix();
}


void ann::add_bias(MatrixXd *x)
{
	x->conservativeResize(x->rows(), x->cols()+1);
    x->col(x->cols()-1).setOnes();

}

void ann::initializeTheta(int inp, int out, MatrixXd *theta)
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
			// distribution.reset();
		}
	}
}

void ann::set_ones_sigd(int nodes, int *num_nodes, int x_col, int y_col)
{
	for (int i = 0; i < nodes; ++i)
	{
		/* code */
	}
}

void ann::learn(MatrixXd X, MatrixXd Y, int nodes, int *num_nodes, bool prelearn)
{
    hidden_layers = new MatrixXd[nodes+1];
    dels = new MatrixXd[nodes+1];
	if (nodes <1)
	{
		fprintf(stderr, "Number of nodes must be greater than 0\n");
	}

	if (!prelearn) //randomly initialize weights
	{
		initializeTheta(num_nodes[0],X.cols()+1, &(hidden_layers[0]));
		dels[0].conservativeResize(num_nodes[0], X.cols()+1);

		for (int i = 1; i < nodes; ++i)
		{
			initializeTheta(num_nodes[i], num_nodes[i-1] + 1, &(hidden_layers[i]));
			dels[0].conservativeResize( num_nodes[i] ,num_nodes[i-1] + 1);

		}

		initializeTheta(Y.cols(), num_nodes[nodes-1], &(hidden_layers[nodes]));

	}//bona

	else{/* stacked autoencoder plz */}

	int i = 0;
	MatrixXd x = normalize(X);
	add_bias(&x);
	int m = x.rows();
	std::cout<< x;

	z_s = new MatrixXd[nodes];
	a_s = new MatrixXd[nodes];
	ones = new MatrixXd[nodes+1];

	set_iterations(1);

	set_ones_sigd(nodes, num_nodes, X.cols(), Y.cols());



	while(i < iter)
	{

		find_grad(x, Y, i, m, nodes);

		i++;

	}




}


void ann::find_grad(MatrixXd X, MatrixXd Y, int i, int m, int nodes)
{

	for (int i = 0; i < m; ++i)
	{
		
		for (int j = 0; j < nodes; ++j)
		{
			z_s[j] = X.row(i)*hidden_layers[i].transpose();
			sigmoid(z_s[j], &(a_s[j]));
			z_s[j].conservativeResize(z_s[j].rows(), z_s[j].cols()+1);
			z_s[j].col(z_s[j].cols()-1).setOnes();

			// x.conservativeResize(x.rows(), x.cols()+1);
			// 		x.col(x.cols()-1).setOnes();

		}




	}















}






















// MatrixXd* ann::gradientStep(MatrixXd x, MatrixXd y, int iter, std::vector<int> nodes, bool bias = 1)
// {
// 	// nodes is a list of the number of nodes we want in a hidden layer

// 	double cost_point[iter];


// 	// Add a bias column to the data if not added
// 	if(!bias)
// 	{
// 		x.conservativeResize(x.rows(), x.cols()+1);
// 		x.col(x.cols()-1).setOnes();
// 	}



// 	int inps = x.cols(); // Number of input nodes
// 	int outp = y.cols(); // Number of final output nodes

// 	int hidden = nodes.size(); // Number of hidden layers

// 	MatrixXd* thetas = new MatrixXd[hidden+1]; // hidden + 2 layers, hidden + 1 connections
// 	MatrixXd* deltas = new MatrixXd[hidden+1];

// 	thetas[0] = initializeTheta(nodes.at(0), inps);

// 	std::cout << hidden << std::endl;

// 	for(int i = 0; i < hidden - 1; ++i)
// 	{
// 		thetas[i+1] = initializeTheta(nodes.at(i+1), nodes.at(i) + 1);
// 	}

// 	thetas[hidden] = initializeTheta(outp, nodes.at(hidden-1) + 1);


// 	for(int i = 0; i < hidden+1; ++i)
// 	{
// 		std::cout << thetas[i] << std::endl;
// 		std::cout << std::endl;
// 	}

// 	int i = 0;
// 	while(i < iter)
// 	{
// 		deltas = nncost(x, y, ++i, thetas, iter, cost_point, 1, hidden+1);
// 		for(int j = 0; j < hidden + 2; ++j){
// 			thetas[j] = thetas[j] - eta*deltas[j];
// 		}

// 		// Length of del's is going to be length of thetas
// 		// for(int j = 0; j < )
// 	}

// 	return thetas;
// }

// MatrixXd* ann::nncost(MatrixXd x, MatrixXd y, int count, MatrixXd* thetas, int iter, double* cost, bool normalize_flag, int theta_len)
// {
// 	double lambda = 5;

// 	std::cout << theta_len << std::endl;

// 	if(normalize_flag)
// 	{
// 		x = normalize(x); // this normalize function assumes the last column is a bias column and thus does not normalize that column
// 	}

// 	// Bias term handled in ann::gradientStep

// 	double this_cost = 0;

// 	MatrixXd a_i(x.cols(), 1);
// 	MatrixXd* a_s = new MatrixXd[theta_len+1]; // keep tabs on this initial size
// 	MatrixXd* z_s = new MatrixXd[theta_len];
// 	MatrixXd* dels = new MatrixXd[theta_len];
// 	MatrixXd* deltas = new MatrixXd[theta_len];
// 	MatrixXd* final_delta = new MatrixXd[theta_len];
// 	MatrixXd z2, a2, h_theta, y_temp, z_i, del_prev, del_i;
	
// 	int h_theta_len;

// 	double log_1, log_2, term_1, term_2;

// 	for(int i = 0; i < x.rows(); ++i)
// 	{
// 		a_s[0] = x.row(i);

// 		std::cout << a_s[0].cols() << std::endl << thetas[0].transpose().rows() << std::endl;

// 		for(int j = 0; j < theta_len; ++j)
// 		{

// 			std::cout << a_s[j]*thetas[j].transpose() << std::endl;

// 			z_s[j] = a_s[j]*thetas[j].transpose();

// 			std::cout << sigmoid(z_s[j]) << std::endl;

// 			a_s[j+1] = sigmoid(z_s[j]);

// 			if(j != theta_len - 1)
// 			{	
// 				a_s[j+1].conservativeResize(a_s[j+1].rows(), a_s[j+1].cols()+1);
// 				a_s[j+1](a_s[j+1].cols()-1) = 1;
// 			}


// 			// z_s[j] = z2;
// 			// a2 = sigmoid(z2);
// 			// a_s[j+1] << 1, a2;

// 		}

// 		std::cout << "here me" << std::endl;

// 		h_theta = a_s[theta_len];

// 		std::cout << h_theta << std::endl;


// 		h_theta_len = thetas[theta_len - 1].rows();

// 		std::cout << h_theta_len << std::endl;
		
// 		y_temp = y.row(i);

// 		std::cout << y_temp << std::endl;

// 		if(count == iter - 1)
// 		{
// 			std::cout << h_theta << std::endl;
// 			std::cout << y_temp << std::endl;
// 		}

// 		for(int j = 0; j < h_theta_len; ++j)
// 		{
// 			log_1 = log(h_theta(j));
// 			log_2 = log(1-h_theta(j));
// 			term_1 = -log_1*y_temp(j);
// 			term_2 = log_2*(1-y_temp(j));
// 			this_cost += term_1 + term_2;

// 			std::cout << this_cost << std::endl;
// 		}

// 		// the dels[theta_len - 1] entry
// 		dels[theta_len - 1] = a_s[theta_len] - y.row(i);

// 		std::cout << "dels[theta_len-1]:\n" << dels[theta_len - 1] << std::endl;

// 		for(int j = theta_len - 1; j > 0; --j)
// 		{
// 			// .row(0) may not be general


// 			std::cout << "thetas[j]:\n" << thetas[j] << std::endl;
// 			std::cout << "dels[j]:\n" << dels[j] << std::endl;
// 			std::cout << "mult:\n" << dels[j]*thetas[j] << std::endl;
// 			std::cout << "sigmoidDeriv:\n" << sigmoidDeriv(z_s[j-1]) << std::endl;

			
// 			dels[j-1] = (dels[j]*thetas[j])*sigmoidDeriv(z_s[j-1]);
// 			std::cout << dels[j-1] << std::endl;
// 		}

// 		for(int j = 0; j < theta_len; ++j)
// 		{
// 			deltas[j] = dels[j].transpose()*a_s[j]/theta_len;
// 		}

// 		if(i == 0)
// 		{
// 			final_delta = deltas;
// 		} else {
// 			for(int k = 0; k < theta_len; ++k)
// 			{
// 				final_delta[k] += deltas[k];
// 			}
// 		}

// 	}

// 	cost[count - 1] = this_cost;

// 	return final_delta;
// }


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
	int nodes[3];
	nodes[0] = 5;
	nodes[1] = 6;
	nodes[2] = 9;


	net.learn(x, y, 3, nodes, 0);

	// std::vector<int> nodes;
	// // nodes.push_back(3);
	// // nodes.push_back(2);
	// // // net.gradientStep(x, y, 100, nodes);


	return 0;
}
