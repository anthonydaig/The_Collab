#include <math.h>
#include "Eigen/Dense"
#include <random>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#define gucci (1)
#define MAXX (0)

using namespace Eigen;

class log_reg {

private:
	void update(MatrixXd *THETA, MatrixXd, MatrixXd, double, int);

public:
	MatrixXd THETA;
	MatrixXd y_out;
	MatrixXd y_pred;
	MatrixXd known_y;
	int classify = 0;
	bool theta_set = 0;
	bool bias = 1;
	bool y_set = 0;
	double eta = .1;
	double lambda = 1;
	//switch iter to epochs
	int iter = 1000;
	// switch iter to epochs
	double accuracy = 0;

	log_reg() { }
	void set_learning_param(double learn);
	void set_regularizer(double regular);
	void set_iterations(int num);
	void learn(MatrixXd, MatrixXd, bool normalize_flag = 1, bool bias2 = 1);
	double sigmoid(MatrixXd x, MatrixXd THETA_col);
	void predict(MatrixXd);
	void vectorize_classification(MatrixXd y);
	void accuracy_frac();
	~log_reg() {}
};

//set parameters from default:
void log_reg::set_learning_param(double learn){eta = learn;}
void log_reg::set_regularizer(double regular){lambda = regular;}
void log_reg::set_iterations(int num){iter = num;}
//set parameters from default


//reformat y for accuracy checking
void log_reg::vectorize_classification(MatrixXd y)
{
	known_y.resize(y.rows(),1);
	y_set = 1;
	for (int i = 0; i < y.rows(); ++i)
	{
		for (int j = 0; j < y.cols(); ++j)
		{
			if(y(i,j))
			{known_y(i,0) = j;
				break;}
		}
	}
}
//reformat y


//Load accuracy
void log_reg::accuracy_frac()
{
	if (!y_set)
	{
		fprintf(stderr, "No Y classifications inputted\n");
	}

	else if(known_y.rows() != y_pred.rows())
	{
		fprintf(stderr, "Number of predictions doesn't match inputted number of Y values\n");
	}

	else{
		double count = 0;
		for (int i = 0; i < known_y.rows(); ++i)
		{
			if (known_y(i,0) == y_pred(i,0))
			{
				count+= 1;
			}
		}
		accuracy = count/known_y.rows();

	}
}
//save accuracy

//sigmoid func
double log_reg::sigmoid(MatrixXd x, MatrixXd THETA_col)
{
	if (theta_set)
	{
		return 1./(1. + exp(-1.*((x*THETA_col)(0,0))));	
	}

	else
	{
		fprintf(stderr, "No initialization of THETA\n");
		return -1;
	}
}
//sigmoid func


//do the actual learning
void log_reg::update(MatrixXd *THETA, MatrixXd x, MatrixXd y, double eta, int iter)
{

	int i = 0;

	MatrixXd THETA_grad(x.cols(), y.cols());

	MatrixXd pred(y.cols(),1); 

	while(i < iter){
		// Set THETA_grad to 0 at the beginning of each iteration

		THETA_grad.setZero();

		for (int j = 0; j < x.rows(); ++j)
		{

			for(int l = 0; l < y.cols(); ++l)
			{
				
				pred(l,0) = sigmoid(x.row(j), THETA->col(l));
				
				for (int k = 0; k < x.cols(); ++k)
				{		
					
					THETA_grad(k,l) += (pred(l,0) - y(j,l))*x(j,k);

				}
			}
			
		}

		THETA_grad *= eta/(x.rows());

		for(int l = 0; l < y.cols(); ++l)
		{
			for (int k = 0; k < x.cols(); ++k)
			{
				
				(*THETA)(k,l) = (*THETA)(k,l)*(1 - eta*lambda/x.rows()) - THETA_grad(k,l); 
			
			}
		}
		
		i++;
	}
}
//learned y's


//input for user
void log_reg::learn(MatrixXd x, MatrixXd y, bool normalize_flag, bool bias2)
{
	if (normalize_flag)
	{
		//normalize x in new matrix x_norm
	}

	MatrixXd x_norm = x;
	bias = bias2;

	if (bias)
	{
		//add bias term
		x_norm.conservativeResize(x_norm.rows(), x_norm.cols()+1);
		x_norm.col(x_norm.cols()-1).setOnes();
	}
	theta_set = 1;

	classify = y.cols();
	THETA.resize(x_norm.cols(), classify);
	THETA.setRandom();

	update(&THETA, x_norm, y, eta, iter);

	y_set = 1;
	vectorize_classification(y);

	predict(x);

	accuracy_frac();
}
//should have everything saved and gucci for use


//do actual prediction
void log_reg::predict(MatrixXd x)
{
	y_out.resize(x.rows(), classify);
	y_pred.resize(x.rows(),1);
	double temp;
	MatrixXd x_norm = x;
	if(bias){x_norm.conservativeResize(x_norm.rows(), x_norm.cols()+1);}
	x_norm.col(x_norm.cols()-1).setOnes();

	int classi;
	for (int i = 0; i < x_norm.rows(); ++i)
	{
		temp = -1;
		classi = 0;


		for(int j = 0; j < classify; ++j)
		{
			y_out(i,j) = sigmoid(x_norm.row(i), THETA.col(j));
			if (y_out(i,j) > temp)
			{

				temp = y_out(i,j);
				classi = j;
			}

		}
		y_pred(i,0) = classi;
	}
}
//prediction



//test
int main()
{
	MatrixXd x(150,4);
	x << 	
		5.1, 3.5, 1.4, 0.2, 
		4.9, 3.0, 1.4, 0.2, 
		4.7, 3.2, 1.3, 0.2, 
		4.6, 3.1, 1.5, 0.2, 
		5.0, 3.6, 1.4, 0.2, 
		5.4, 3.9, 1.7, 0.4, 
		4.6, 3.4, 1.4, 0.3, 
		5.0, 3.4, 1.5, 0.2, 
		4.4, 2.9, 1.4, 0.2, 
		4.9, 3.1, 1.5, 0.1, 
		5.4, 3.7, 1.5, 0.2, 
		4.8, 3.4, 1.6, 0.2, 
		4.8, 3.0, 1.4, 0.1, 
		4.3, 3.0, 1.1, 0.1, 
		5.8, 4.0, 1.2, 0.2, 
		5.7, 4.4, 1.5, 0.4, 
		5.4, 3.9, 1.3, 0.4, 
		5.1, 3.5, 1.4, 0.3, 
		5.7, 3.8, 1.7, 0.3, 
		5.1, 3.8, 1.5, 0.3, 
		5.4, 3.4, 1.7, 0.2, 
		5.1, 3.7, 1.5, 0.4, 
		4.6, 3.6, 1.0, 0.2, 
		5.1, 3.3, 1.7, 0.5, 
		4.8, 3.4, 1.9, 0.2, 
		5.0, 3.0, 1.6, 0.2, 
		5.0, 3.4, 1.6, 0.4, 
		5.2, 3.5, 1.5, 0.2, 
		5.2, 3.4, 1.4, 0.2, 
		4.7, 3.2, 1.6, 0.2, 
		4.8, 3.1, 1.6, 0.2, 
		5.4, 3.4, 1.5, 0.4, 
		5.2, 4.1, 1.5, 0.1, 
		5.5, 4.2, 1.4, 0.2, 
		4.9, 3.1, 1.5, 0.1, 
		5.0, 3.2, 1.2, 0.2, 
		5.5, 3.5, 1.3, 0.2, 
		4.9, 3.1, 1.5, 0.1, 
		4.4, 3.0, 1.3, 0.2, 
		5.1, 3.4, 1.5, 0.2, 
		5.0, 3.5, 1.3, 0.3, 
		4.5, 2.3, 1.3, 0.3, 
		4.4, 3.2, 1.3, 0.2, 
		5.0, 3.5, 1.6, 0.6, 
		5.1, 3.8, 1.9, 0.4, 
		4.8, 3.0, 1.4, 0.3, 
		5.1, 3.8, 1.6, 0.2, 
		4.6, 3.2, 1.4, 0.2, 
		5.3, 3.7, 1.5, 0.2, 
		5.0, 3.3, 1.4, 0.2, 
		7.0, 3.2, 4.7, 1.4, 
		6.4, 3.2, 4.5, 1.5, 
		6.9, 3.1, 4.9, 1.5, 
		5.5, 2.3, 4.0, 1.3, 
		6.5, 2.8, 4.6, 1.5, 
		5.7, 2.8, 4.5, 1.3, 
		6.3, 3.3, 4.7, 1.6, 
		4.9, 2.4, 3.3, 1.0, 
		6.6, 2.9, 4.6, 1.3, 
		5.2, 2.7, 3.9, 1.4, 
		5.0, 2.0, 3.5, 1.0, 
		5.9, 3.0, 4.2, 1.5, 
		6.0, 2.2, 4.0, 1.0, 
		6.1, 2.9, 4.7, 1.4, 
		5.6, 2.9, 3.6, 1.3, 
		6.7, 3.1, 4.4, 1.4, 
		5.6, 3.0, 4.5, 1.5, 
		5.8, 2.7, 4.1, 1.0, 
		6.2, 2.2, 4.5, 1.5, 
		5.6, 2.5, 3.9, 1.1, 
		5.9, 3.2, 4.8, 1.8, 
		6.1, 2.8, 4.0, 1.3, 
		6.3, 2.5, 4.9, 1.5, 
		6.1, 2.8, 4.7, 1.2, 
		6.4, 2.9, 4.3, 1.3, 
		6.6, 3.0, 4.4, 1.4, 
		6.8, 2.8, 4.8, 1.4, 
		6.7, 3.0, 5.0, 1.7, 
		6.0, 2.9, 4.5, 1.5, 
		5.7, 2.6, 3.5, 1.0, 
		5.5, 2.4, 3.8, 1.1, 
		5.5, 2.4, 3.7, 1.0, 
		5.8, 2.7, 3.9, 1.2, 
		6.0, 2.7, 5.1, 1.6, 
		5.4, 3.0, 4.5, 1.5, 
		6.0, 3.4, 4.5, 1.6, 
		6.7, 3.1, 4.7, 1.5, 
		6.3, 2.3, 4.4, 1.3, 
		5.6, 3.0, 4.1, 1.3, 
		5.5, 2.5, 4.0, 1.3, 
		5.5, 2.6, 4.4, 1.2, 
		6.1, 3.0, 4.6, 1.4, 
		5.8, 2.6, 4.0, 1.2, 
		5.0, 2.3, 3.3, 1.0, 
		5.6, 2.7, 4.2, 1.3, 
		5.7, 3.0, 4.2, 1.2, 
		5.7, 2.9, 4.2, 1.3, 
		6.2, 2.9, 4.3, 1.3, 
		5.1, 2.5, 3.0, 1.1, 
		5.7, 2.8, 4.1, 1.3, 
		6.3, 3.3, 6.0, 2.5, 
		5.8, 2.7, 5.1, 1.9, 
		7.1, 3.0, 5.9, 2.1, 
		6.3, 2.9, 5.6, 1.8, 
		6.5, 3.0, 5.8, 2.2, 
		7.6, 3.0, 6.6, 2.1, 
		4.9, 2.5, 4.5, 1.7, 
		7.3, 2.9, 6.3, 1.8, 
		6.7, 2.5, 5.8, 1.8, 
		7.2, 3.6, 6.1, 2.5, 
		6.5, 3.2, 5.1, 2.0, 
		6.4, 2.7, 5.3, 1.9, 
		6.8, 3.0, 5.5, 2.1, 
		5.7, 2.5, 5.0, 2.0, 
		5.8, 2.8, 5.1, 2.4, 
		6.4, 3.2, 5.3, 2.3, 
		6.5, 3.0, 5.5, 1.8, 
		7.7, 3.8, 6.7, 2.2, 
		7.7, 2.6, 6.9, 2.3, 
		6.0, 2.2, 5.0, 1.5, 
		6.9, 3.2, 5.7, 2.3, 
		5.6, 2.8, 4.9, 2.0, 
		7.7, 2.8, 6.7, 2.0, 
		6.3, 2.7, 4.9, 1.8, 
		6.7, 3.3, 5.7, 2.1, 
		7.2, 3.2, 6.0, 1.8, 
		6.2, 2.8, 4.8, 1.8, 
		6.1, 3.0, 4.9, 1.8, 
		6.4, 2.8, 5.6, 2.1, 
		7.2, 3.0, 5.8, 1.6, 
		7.4, 2.8, 6.1, 1.9, 
		7.9, 3.8, 6.4, 2.0, 
		6.4, 2.8, 5.6, 2.2, 
		6.3, 2.8, 5.1, 1.5, 
		6.1, 2.6, 5.6, 1.4, 
		7.7, 3.0, 6.1, 2.3, 
		6.3, 3.4, 5.6, 2.4, 
		6.4, 3.1, 5.5, 1.8, 
		6.0, 3.0, 4.8, 1.8, 
		6.9, 3.1, 5.4, 2.1, 
		6.7, 3.1, 5.6, 2.4, 
		6.9, 3.1, 5.1, 2.3, 
		5.8, 2.7, 5.1, 1.9, 
		6.8, 3.2, 5.9, 2.3, 
		6.7, 3.3, 5.7, 2.5, 
		6.7, 3.0, 5.2, 2.3, 
		6.3, 2.5, 5.0, 1.9, 
		6.5, 3.0, 5.2, 2.0, 
		6.2, 3.4, 5.4, 2.3, 
		5.9, 3.0, 5.1, 1.8; 
	MatrixXd y(150, 3);
	y << 
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		gucci, MAXX, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, gucci, MAXX,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci,
		MAXX, MAXX, gucci;

	log_reg logg; 
	// logg.set_iterations(100000);

	// logg.set_regularizer(0);

	logg.learn(x, y, 0, 1);

	logg.predict(x);

	

	printf("%d\n",logg.iter );

	logg.vectorize_classification(y);


	// std::cout << "THIS IS THETA: \n" << logg.THETA << std::endl;

	// std::cout << "THIS IS y_out \n" << logg.y_out << std::endl;

	std::cout << "THIS IS y_pred \n" << logg.y_pred << std::endl;

	std::cout << "THIS IS y_known\n" << logg.known_y << std::endl;

	std::cout << "ACCURACY: \n" << logg.accuracy << std::endl;

	return 1;
}
//test






