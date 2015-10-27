#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "Eigen/Dense"

#define gucci (1)
#define MAXX (0)


using namespace Eigen;


class knn {

private:
	double dist(int, MatrixXd, MatrixXd);
	

public:
	MatrixXd Neighbors;
	MatrixXd Classifications;

	void learn(MatrixXd X, MatrixXd Y, MatrixXd test, int k = -1);
	//lets add an option that lets us ADD data points without running through everything again
	// i.e. heapsort the shit out of this baby

	
};



double knn::dist(int m, MatrixXd pointa, MatrixXd pointb) {


	double distance_sq = 0.0;
	int i = 0;
	for (; i < m; i++) {
		distance_sq = distance_sq + (pointa(0,i)-pointb(0,i))*(pointa(0,i)-pointb(0,i));
	}

	return distance_sq;
}


void knn::learn(MatrixXd X, MatrixXd Y, MatrixXd test, int k/* also add option to define distance*/)
{
	if (k == -1)
	{
		k = Y.cols();
	}

	Neighbors.resize(test.rows(), X.rows());

	for (int i = 0; i < test.rows(); ++i)
	{
		for (int j = 0; j < X.rows(); ++j)
		{
			Neighbors(i,j) = dist(test.cols(), test.row(i), X.row(j));
		}

		

	}


	std::cout << Neighbors << std::endl;


}




int main()
{
	MatrixXd x(149,4);
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
		5.9, 3.0, 5.1, 1.8; 
	MatrixXd y(149, 3);
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
		MAXX, MAXX, gucci;



	MatrixXd test(5, 4);
	test << 
	.1, .2, .3, .4, 
	.4, .3, .2, .1, 
	.1, .1, .1, .1,
	.2, .2, .2, .2, 
	.4, .5, .6, .7;
	// log_reg logg; 

	knn knnn;

	knnn.learn(x, y, test, 7);


	return 1;
}