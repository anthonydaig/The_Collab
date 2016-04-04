#include <math.h>
#include "Eigen/Dense"
#include <random>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>


using namespace Eigen;

class ann {

private:
	
	
	void sigmoid(MatrixXd, MatrixXd*);
	void sigmoidDeriv(MatrixXd*,int);
	void initializeTheta(int, int, MatrixXd*);
	void add_bias(MatrixXd*);
	void set_ones_sigd(int nodes, int *num_nodes, int x_col, int y_col);
	void multiply_two(MatrixXd x1, MatrixXd *x2, int);
	MatrixXd *z_s;
	MatrixXd *a_s;
	MatrixXd *ones;
	MatrixXd *assoc_err;
	MatrixXd *gradients;
	MatrixXd *momenta;
	MatrixXd *temp_mom;
	int layers;
	void find_grad(MatrixXd X, MatrixXd Y, int m, int nodes, double *cost);

public:
	double eta = 0.1;
	double lambda = .015;
	double momentum = .5;
	int iter = 1000;
	MatrixXd *hidden_layers;

	ann() { }
	MatrixXd normalize(MatrixXd);
	void set_learning_param(double eta1){eta = eta1;}
	void set_regularizer(double regular){lambda = regular;}
	void set_iterations(int num){iter = num;}
	void set_iterations(double momentum1){momentum = momentum1;}
	void clean();
	// MatrixXd* gradientStep(MatrixXd, MatrixXd, int, std::vector<int>, bool);
	void learn(MatrixXd, MatrixXd, int, int*, bool prelearn =1/*whether or not we want stacked autoencoders (default 1)*/);

	MatrixXd predict(MatrixXd);

};

void ann::clean()
{
	delete[] z_s;
	delete[] a_s; 
	delete[] ones;
	delete[] assoc_err;
	delete[] gradients;
	delete[] momenta;
	delete[] temp_mom;
}

MatrixXd ann::normalize(MatrixXd x)
{
	double mean, sd;
	for(int i = 0; i < x.cols(); ++i)
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

void ann::sigmoidDeriv(MatrixXd *x1, int m)
{
	double temp;
	for (int i = 0; i < m; ++i)
	{
		temp = (1./(1. + exp(-1. * ((*x1)(0,i)))));
		(*x1)(0,i) = (1 - temp)*temp;
	}
}

void ann::multiply_two(MatrixXd x1, MatrixXd *x2, int m)
{
	for (int i = 0; i < m; ++i)
	{
		(*x2)(0,i) = (*x2)(0,i)*x1(0,i);
	}
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
		}
	}
}

void ann::set_ones_sigd(int nodes, int *num_nodes, int x_col, int y_col)
{
	ones[0] = MatrixXd::Ones(num_nodes[0], x_col+1);
	for (int i = 1; i < nodes; ++i)
	{
		ones[i] = MatrixXd::Ones(num_nodes[i], num_nodes[i-1]+1);
	}
	ones[nodes] = MatrixXd::Ones(y_col, num_nodes[nodes-1]);
}

void ann::learn(MatrixXd X, MatrixXd Y, int nodes, int *num_nodes, bool prelearn)
{
	layers = nodes;
    hidden_layers = new MatrixXd[nodes+1];
	if (nodes <1)
	{
		fprintf(stderr, "Number of nodes must be greater than 0\n");
	}

	if (!prelearn) //randomly initialize weights
	{
		initializeTheta(num_nodes[0],X.cols()+1, &(hidden_layers[0]));

		for (int i = 1; i < nodes; ++i)
		{
			initializeTheta(num_nodes[i], num_nodes[i-1] + 1, &(hidden_layers[i]));
		}

		initializeTheta(Y.cols(), num_nodes[nodes-1] + 1, &(hidden_layers[nodes]));

	}

	else{/* stacked autoencoder plz */}

	int i = 0;
	MatrixXd x = normalize(X);
	add_bias(&x);
	int m = x.rows();

	z_s = new MatrixXd[nodes+1];
	a_s = new MatrixXd[nodes+2];
	ones = new MatrixXd[nodes+1];
	assoc_err = new MatrixXd[nodes+1];
	gradients = new MatrixXd[nodes+1];
	momenta = new MatrixXd[nodes+1];
	temp_mom = new MatrixXd[nodes+1];

	set_iterations(1000);

	set_ones_sigd(nodes, num_nodes, X.cols(), Y.cols());

	double cost = 0;


	while(i < iter)
	{
		find_grad(x, Y, m, nodes, &cost);

		for (int j = 0; j < nodes+1; ++j)
		{
			if (i)
			{
				gradients[j] = (gradients[j] + lambda*hidden_layers[j])/m;
				temp_mom[j] = -eta*gradients[j] + momentum*momenta[j];
				hidden_layers[j] = hidden_layers[j] + temp_mom[j];
				momenta[j] = temp_mom[j];
			}

			else
			{
				gradients[j] = (gradients[j] + lambda*hidden_layers[j])/m;
				momenta[j] = -eta*gradients[j];
				hidden_layers[j] = hidden_layers[j] + momenta[j];
			}

		}
		
		cost = 0;
		i++;

	}
}

void ann::find_grad(MatrixXd X, MatrixXd Y, int m, int nodes, double *cost)
{
	int j;

	for (int i = 0; i < m; ++i)
	{
		a_s[0] = X.row(i);

		for (j = 0; j < nodes+1; ++j)
		{
		
			z_s[j] = a_s[j]*(hidden_layers[j].transpose());

			sigmoid(z_s[j], &(a_s[j+1]));

			a_s[j+1].conservativeResize(a_s[j+1].rows(), a_s[j+1].cols()+1);

			a_s[j+1].col(a_s[j+1].cols()-1).setOnes();

		}

		a_s[nodes+1].conservativeResize(a_s[nodes+1].rows(), a_s[nodes+1].cols()-1);

	

		for (j = 0; j < Y.cols(); ++j)
		{
			*cost += -1 * Y(i,j)*log(a_s[nodes+1](0,j))  
			- (1. - Y(i,j))*log(1. - a_s[nodes+1](0,j));

		}

		assoc_err[nodes] = a_s[nodes+1] - Y.row(i);

		for (j = nodes; j > 0; --j)
		{
			assoc_err[j-1] = assoc_err[j]*(hidden_layers[j]);
			assoc_err[j-1].conservativeResize(1, assoc_err[j-1].cols()-1);
			sigmoidDeriv(&(z_s[j-1]), z_s[j-1].cols());
			multiply_two(z_s[j-1], &(assoc_err[j-1]), assoc_err[j-1].cols());
		}

		for (int j = 0; j < nodes+1; ++j)
		{			
			if (i)
			{
				gradients[j] = gradients[j] + (assoc_err[j].transpose() * a_s[j]);
			}
			else
			{
				gradients[j] = (assoc_err[j].transpose() * a_s[j]);
			}	
		}
	}

	*cost = *cost / m;

}


MatrixXd ann::predict(MatrixXd unkn)
{
	MatrixXd inp = normalize(unkn);
	add_bias(&inp);
	MatrixXd predictions(unkn.rows(), hidden_layers[layers].rows());


	for (int i = 0; i < unkn.rows(); ++i)
	{
	

		a_s[0] = inp.row(i);

		for (int j = 0; j < layers+1; ++j)
		{

			z_s[j] = a_s[j]*(hidden_layers[j].transpose());

			sigmoid(z_s[j], &(a_s[j+1]));
			a_s[j+1].conservativeResize(a_s[j+1].rows(), a_s[j+1].cols()+1);
			a_s[j+1].col(a_s[j+1].cols()-1).setOnes();

		}

		for (int j = 0; j < hidden_layers[layers].rows(); ++j)
		{
			predictions(i,j) = a_s[layers+1](0,j);

		}

	}

	return predictions;

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

	int nodes[3];
	nodes[0] = 5;
	nodes[1] = 6;
	nodes[2] = 9;

	// net.enc_dec(x, 2);


	// net.learn(x, y, 3, nodes, 0);

	// std::cout << net.predict(x) << "\nsuppp\n"<< std::endl;
	// std::cout << x << std::endl;


	return 0;
}
