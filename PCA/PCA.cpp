#include <math.h>
#include "Eigen/Dense"
#include <random>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

using namespace Eigen;

class PCA {


private:
	double total_eig = 0; 
	int features;

public:
	MatrixXd Eigen_Vectors;
	MatrixXd Basis; 
	double *Eigen_Values;
	PCA() { }
	void get_eigen(MatrixXd);
	void pick_basis(double);
};


void PCA::get_eigen(MatrixXd X)
{
	MatrixXd centered = X.rowwise() - X.colwise().mean();
	MatrixXd cov = centered.adjoint() * centered;
	SelfAdjointEigenSolver<MatrixXd> eig(cov);


	Eigen_Vectors = eig.eigenvectors();

	Eigen_Values = new double[X.cols()];
	features = X.cols();

	for (int i = 0; i < X.cols(); ++i)
	{
		Eigen_Values[i] = (eig.eigenvalues()(i,0));
		total_eig += Eigen_Values[i];
	}
}

void PCA::pick_basis(double eps)
{
	double variance = 0;
	int i = 0;
	while(variance <= eps)
	{
		variance += Eigen_Values[i] / total_eig;
		std::cout << Eigen_Values[i] / total_eig  << std::endl;
		std::cout<< variance << "\n"  << std::endl;
		i++;
	}
	Basis = Eigen_Vectors.rightCols(features - i + 1);
}
