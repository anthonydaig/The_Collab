#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>
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

using namespace Eigen; 

typedef struct node Node;
typedef struct connection Connection;
struct node{
	int activate;
	double fire_time;
};

double why(double t, int i)
{
	double tau = 7;
	double help = t/tau;
	if (t < 0)
	{
		return 0;
	}
	else
	{
		double k;
		if (i == 0){k = -1;}
		else{k=1;}
		return k*help*exp(1-help);
	}
}

double dwhy(double t, int i)
{
	double tau = 7;
	double help = t/tau;
	if (t < 0)
	{
		return 0;
	}
	else
	{
		double k;
		if (i == 0){k = -1;}
		else{k=1;}
		return k*exp(1-help)*(tau - t)/(tau*tau);
	}
}

double sum_through(Node node_prev, int i, int j, double ***weights, int k, double t)
{
	double dt = 100;
	double help = 0;

	for (int m = 0; m < k; ++m)
	{
		help += weights[i][j][m] * why(t-node_prev.fire_time-m, i);
	}
	return help;
}

double feed_one(Node **network, double ***weights_1, double ***weights_2,
 MatrixXd x, int inp_node, int hidden_node, int output_node, int k, MatrixXd y, int max)
{
	//set up input nodes:
	for (int i = 0; i < x.cols(); ++i)
	{
		network[0][i+1].fire_time = (double) x(i);
	}
	network[0][0].fire_time = 0;

	// set up input nodes

	double t = 0;

	double time_step = .01;
	double check;
	double thres = 3;
	int flag = 0;

	while(t<= max)
	{
		for (int i = 0; i < inp_node; ++i)
		{
			if (network[0][i].fire_time == t)
			{
				network[0][i].activate = 1;
			}
		}




		for (int i = 0; i < hidden_node; ++i)
		{
			if(!network[1][i].activate)
			{
				check = 0;
				for (int j = 0; j < inp_node; ++j)
				{
					
					if (network[0][j].activate)
					{
						check += sum_through(network[0][j], j, i, weights_1, k, t);
					}
				}

				if (check >= thres)
				{
					network[1][i].activate = 1;
					network[1][i].fire_time = t;
				}
			}
		}



		for (int i = 0; i < output_node; ++i)
		{
			if (!network[2][i].activate)
			{

				check = 0;
				for (int j = 0; j < hidden_node/*-1*/; ++j)
				{
					if (network[1][j].activate)
					{
						check += sum_through(network[1][j],j,i,weights_2, k, t);
					}
				}

			/*	if(network[1][hidden_node-1].activate)
				{
					check -= sum_through(network[1][hidden_node-1],hidden_node-1, i, weights_2, k, t);
				}
*/

				if(check>= thres)
				{
					network[2][i].activate = 1;
					network[2][i].fire_time = t;
				}

			}


		}
		t+= time_step;

}


	// find the cost now: 
	double cost = 0;
	double temp;
	for (int i = 0; i < output_node; ++i)
	{
		temp = (network[2][i].fire_time - y(i));
		cost += .5 * temp * temp;
	}
	return cost;
}

void reset_network(Node **network, int inp, int hidden, int output, int max)
{
	for (int i = 0; i < inp; ++i)
	{
		network[0][i].fire_time = 0;
		network[0][i].activate = 0;
	}
	for (int i = 0; i < hidden; ++i)
	{
		network[1][i].fire_time = max;
		network[1][i].activate = 0;
	}

	for (int i = 0; i < output; ++i)
	{
		network[2][i].fire_time = max;
		network[2][i].activate = 0;
	}


}


double sum_through_deriv(Node node, int i, int j, double ***weights, int t_j, int k, double dt)
{
	double help =0;
	double index; 
	// printf("node.firetime: %f\n", node.fire_time);

	for (int l = 0; l < k; ++l)
	{
		index = t_j - node.fire_time - l;
		// std::cout << index << std::endl;
		// printf("weight: %f, %i, %f\n", weights[i][j][l], t_j, node.fire_time);
		help += weights[i][j][l] * dwhy(index, i);
	}

	return help;
}

double *back_prop_1(Node **network, double ***weights_2, MatrixXd y, 
	int hidden_node, int output_node, int k, double dt)
{
	double helpp; 
	double *del = new double[output_node]; 
	double help;
	// printf("k: %i\n", k);
	for (int j = 0; j < output_node; ++j)
	{
		help = 0;
		for (int i = 0; i < hidden_node; ++i)
		{
			// printf("i: %i, j: %i, fire_time: %f\n", i, j, network[2][j].fire_time);
			helpp = sum_through_deriv(network[1][i], i, j, weights_2, network[2][j].fire_time,
				k, dt);
			// printf("helpp: %f\n", helpp);
			// printf("this connection: %f, index: %i, %i\n",helpp, i, j );
			help += helpp;
			
		}
		
		del[j] = (y(j) - network[2][j].fire_time)/help;

	}
	// printf("\n\n\n\n");
	return del;
}


double *back_prop_2(Node **network, double ***weights_1, double ***weights_2, double *del,
	int input_node, int hidden_node, int output_node, int k, double dt)
{
	double help_num;
	double help_den;
	double *del2 = new double[hidden_node];


	for (int i = 0; i < hidden_node; ++i)
	{
		help_num = 0;
		for (int j = 0; j < output_node; ++j)
		{
			help_num += del[j] * sum_through_deriv(network[1][i], i, j, 
				weights_2, network[2][j].fire_time, k, dt);
		}

		for (int h = 0; h < input_node; ++h)
		{
			help_den += sum_through_deriv(network[0][h], h, i,
				weights_1, network[1][i].fire_time, k, dt);
		}

		del2[i] = help_num/help_den;

	}

	return del2;
}

void print_network(Node **network, int inp_node, int hidden_node, int output_node)
{
	for (int i = 0; i < inp_node; ++i)
	{
		printf("%f 		", network[0][i].fire_time);
	}

	std::cout<<"\n"<<std::endl;

	for (int i = 0; i < hidden_node; ++i)
	{
		printf("%f 	", network[1][i].fire_time);
	}

		std::cout<<"\n"<<std::endl;

	for (int i = 0; i < output_node; ++i)
	{
		printf("		%f 		", network[2][i].fire_time);
	}
	std::cout<<"\n"<<std::endl;

}

int main()
{
	int score;
	const int inp_node = 3;
	const int hidden_node = 5;
	const int output_node = 1;
	const int k = 15;
	// const int k = max - 1;
	const double dt = 100;
	const int max = 16;

	Node *inp_layer = new Node[inp_node];
	Node *hidden_layer = new Node[hidden_node];
	Node *output_layer = new Node[output_node];
	Node **network = new Node*[3];
	double learning_param = .015;
	network[0] = inp_layer;
	network[1] = hidden_layer;
	network[2] = output_layer;

	for (int i = 0; i < inp_node; ++i)
	{
		inp_layer[i].fire_time = max;
		inp_layer[i].activate = 0;
	}

	for (int i = 0; i < hidden_node; ++i)
	{
		hidden_layer[i].fire_time = max;
		hidden_layer[i].activate = 0;
	}

	for (int i = 0; i < output_node; ++i)
	{
		output_layer[i].fire_time = max;
		output_layer[i].activate = 0;
	}

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution_noise(1,4);
	std::uniform_real_distribution<double> distribution_noise_out(1,4);
		//initialize inputs (for XOR)
	MatrixXd x(4,2);
	x<< 0,0,
		0,6,
		6,0,
		6,6;

	//initialize outputs (for XOR)
	MatrixXd y(4,1);
	y<< 16,
		10,
		10,
		16;

	double ***weights_1 = new double**[inp_node];
	double ***temp1 = new double**[inp_node];
	for (int i = 0; i < inp_node; ++i)
	{
		weights_1[i] = new double*[hidden_node];
		temp1[i] = new double*[hidden_node];

		for (int j = 0; j < hidden_node; ++j)
		{
			temp1[i][j] = new double[k];
			weights_1[i][j] = new double[k];
			for (int m = 0; m < k; ++m)
			{
				temp1[i][j][m] = 0;
				weights_1[i][j][m] = distribution_noise(generator);
			}
		}
	}
	double ***weights_2 = new double**[hidden_node];
	double ***temp2 = new double**[hidden_node];
	for (int i = 0; i < hidden_node; ++i)
	{
		temp2[i] = new double*[output_node];
		weights_2[i] = new double*[output_node];

		for (int j = 0; j < output_node; ++j)
		{
			temp2[i][j] = new double[k];
			weights_2[i][j] = new double[k];

			for (int m = 0; m < k; ++m)
			{
				temp2[i][j][m] = 0;
				weights_2[i][j][m] = distribution_noise_out(generator);
			}
		}
	}
	double *del1;
	double *del2; 
	int step =0;
	int iter = 20000;
	double cost; 

	while(step < iter)
	{
		cost = 0;


		for (int tr = 0; tr < x.rows(); ++tr)
		{

			cost += feed_one(network, weights_1, weights_2, x.row(tr), 
				inp_node, hidden_node, output_node, k, y.row(tr), max);

			del1 = back_prop_1(network, weights_2, y.row(tr), 
				hidden_node, output_node, k, dt);


			del2 = back_prop_2(network, weights_1, weights_2, del1,
			inp_node, hidden_node, output_node, k, dt);

			if (step%100 == 0)
				{print_network(network, inp_node, hidden_node, output_node);
					std::cout << "\n" << std::endl;
					std::cout << "firetime: " << network[2][0].fire_time << std::endl;

					/*std::cin.get();*/}


			for (int i = 0; i < hidden_node; ++i)
			{
				for (int j = 0; j < output_node; ++j)
				{
					for (int l = 0; l < k; ++l)
					{
						temp2[i][j][l] += -1*learning_param*
						why(network[2][j].fire_time - network[1][i].fire_time - l, i)
						*del1[j]/x.rows();
					}
				}
			}

			for (int h = 0; h < inp_node; ++h)
			{
				for (int i = 0; i < hidden_node; ++i)
				{
					for (int l = 0; l < k; ++l)
					{
						// temp1[h][i][l] = 4;
						temp1[h][i][l] += -1*learning_param
						*why(network[1][i].fire_time - network[0][h].fire_time - l, 1)
						*del2[i]/x.rows();
					}
				}
			}


			reset_network(network, inp_node, hidden_node, output_node, max);

		}



		for (int i = 0; i < hidden_node; ++i)
		{
			for (int j = 0; j < output_node; ++j)
			{
				for (int l = 0; l < k; ++l)
				{
					// printf("%f\n", temp2[i][j][l]);
					weights_2[i][j][l] += temp2[i][j][l];
					temp2[i][j][l] = 0;
					weights_2[i][j][l] = weights_2[i][j][l];
 				}
			}
		}



		for (int i = 0; i < inp_node; ++i)
		{
			for (int j = 0; j < hidden_node; ++j)
			{
				for (int l = 0; l < k; ++l)
				{
					weights_1[i][j][l] += temp1[i][j][l];
					temp1[i][j][l] = 0;
					weights_1[i][j][l] = std::abs(weights_1[i][j][l]);
				}
			}
		}

		

		// print_network(network, inp_node, hidden_node, output_node);

		// return 0;

			if (step%100 == 0){//print_network(network, inp_node, hidden_node, output_node);
			// printf("\n\n\n\n");

			// for (int i = 0; i < hidden_node; ++i)
			// {
			// 	printf("\n");
			// 	for (int j = 0;j < output_node; ++j)
			// 	{
			// 		for (int m = 0; m < k; ++m)
			// 		{
			// 			printf("weights_2[%i][%i][%i] = %f ",i,j,m,weights_2[i][j][m]);
			// 		}
			// 		printf("\n");
			// 	}
			// }

			printf("\n\n\n");


			std::cout << "Cost: " << cost/x.rows() << std::endl;
			// std::cin.get();
			}

		step++;
	}





	delete[] inp_layer;
	delete[] hidden_layer;
	delete[] output_layer;
	delete[] network;

	for (int i = 0; i < inp_node; ++i)
	{
		for (int j = 0; j < hidden_node; ++j)
		{
			delete[] weights_1[i][j];
		}
		delete[] weights_1[i];
	}

	delete[] weights_1;


	for (int i = 0; i < hidden_node; ++i)
	{
		for (int j = 0; j < output_node; ++j)
		{
			delete[] weights_2[i][j];
		}
		delete[] weights_2[i];
	}

	delete[] weights_2;





}




