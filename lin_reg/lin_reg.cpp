#include <math.h>
#include "Eigen/Dense"
#include <random>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

using namespace Eigen;	

class lin_reg{

private:

	void linsolv(MatrixXd a, int n, VectorXd y);





public:
	MatrixXd THETA;
	bool bias = 1;
	lin_reg(){ }
	void learn(MatrixXd x, MatrixXd y, bool normalize_flag = 1, bool bias2 = 1);
	~lin_reg(){}

};


void lin_reg::linsolv(MatrixXd a, int n, VectorXd y)
{

	double c;
	double s;
	double r;

	double up;
	double low;

	double temp_up = 0;
	double temp_down = 0;


	double *aa = new double[n*n];
	double *yy = new double[n];


	for (int i = 0; i < n*n; ++i)
	{
		aa[i] = a(i);
	}
	for (int i = 0; i < n; ++i)
	{
		yy[i] = y(i);
	}

	for (int i = 0; i < n-1; ++i)
	{



		for (int k = 1; k+i < n; ++k)
		{
			up = aa[n*i+i];
			low = aa[(i+k)*n+i];
			r = sqrt(up*up + low*low);

			if (!r)
			{
				continue;
			}


			c = up/r;
			s = -low/r;

			for (int j = i; j < n; ++j)
			{
				temp_up = c*aa[i*n+j] - s*aa[(i+k)*n+j];
				temp_down = s*aa[i*n+j] + c*aa[(i+k)*n+j];

				aa[i*n + j] = temp_up;
				aa[(i+k)*n+j] = temp_down;

			}

			temp_up = c*yy[i] - s*yy[i+k];
			temp_down = s*yy[i] + c*yy[i+k];

			yy[i] = temp_up;
			yy[i+k] = temp_down;

		}


	}

	double temp;
	for (int i = n-1; i >= 0; --i)
	{
		temp = 0;
		
		for (int j = i+1; j < n; ++j)
		{
			temp += THETA(j,0)*aa[i*n+j];
		}

		THETA(i,0) = (yy[i] - temp)/aa[i*n+i];

	}

	free(yy);
	free(aa);

}



void lin_reg::learn(MatrixXd x, MatrixXd y, bool normalize_flag, bool bias2)
{
	MatrixXd x_norm = x;

	bias = bias2;

	if (normalize_flag)
	{
		//normalize x in new matrix x_norm
	}
	if (bias)
	{
		//add bias term
		x_norm.conservativeResize(x_norm.rows(), x_norm.cols()+1);
		x_norm.col(x_norm.cols()-1).setOnes();
	}

	THETA.resize(x_norm.cols(),1);

	MatrixXd y_help;
	y_help.resize(y.rows(),1);

	y_help = x_norm.transpose()*y;

	x_norm = x_norm.transpose()*x_norm;

	linsolv(x_norm, x_norm.rows(), y_help);


}


int main()
{
	return 1;

}













