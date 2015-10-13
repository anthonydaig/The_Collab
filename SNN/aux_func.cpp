#include "aux_func.h"
using namespace Eigen;

double dudt(double u,double v, double w)
{
	return OOEPS1 * (u - u*u + w*(Q-u))/BETA;
}

double dvdt(double u, double v, double w)
{
	return (u-v)/BETA;
}

double dwdt(double u,double v, double w)
{
	return OOEPS2 * (PHI + F * v - w * (u + Q))/BETA;
}


double add_it(Node *node, int j, int i, double **weights)
{
	return weights[j][i]*node->w;
}

void update_node(Node *node)
{
	double help1, help2, help3;
	help1 = DT * dudt(node->u, node->v, node->w);
	help2 = DT * dvdt(node->u, node->v, node->w);
	help3 = DT * dwdt(node->u, node->v, node->w);
	node->u += help1;
	node->v += help2; 
	node->w += help3;
}

void print_weights(double **weights, int prev, int next)
{
	for (int i = 0; i < prev; ++i)
	{
		for (int j = 0; j < next; ++j)
		{
			printf("%f, ", weights[i][j]);
		}
		printf("\n");
	}
}

void update(Node **network, int inp_node, int hidden_node, int output_node)
{
	for (int i = 0; i < inp_node; ++i)
	{
		if (network[0][i].activate)
		{
			update_node(&network[0][i]);
		}
	}
	for (int i = 0; i < hidden_node; ++i)
	{
		if (network[1][i].activate)
		{

			update_node(&network[1][i]);
			
		}
	}
	for (int i = 0; i < output_node; ++i)
	{
		if (network[2][i].activate)
		{
			update_node(&network[2][i]);
		}
	}
}


double feed_one(Node **network, double **weights_1, double **weights_2,
 MatrixXd x, int inp_node, int hidden_node, int output_node, MatrixXd y, int max)
{
	//set up input nodes:
	for (int i = 0; i < x.cols(); ++i)
	{
		network[0][i+1].fire_time = (double) x(i);
	}
	network[0][0].fire_time = 0;

	// set up input nodes

	double t = 0;
	double time_step = DT;
	double check;
	double thres = 80/ALPHA;
	int flag = 0;

	while(t<= max)
	{
		

		for (int i = 0; i < inp_node; ++i)
		{
			if ((!network[0][i].activate) &&
				network[0][i].fire_time <= t)
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
						check += add_it(&(network[0][j]), j, i, weights_1);
					}
				}
				if (check/ALPHA >= thres)
				{	
					network[1][i].activate = 1;
					network[1][i].fire_time = t;
					for (int j = 0; j < inp_node; ++j)
					{
						network[0][j].deriv_fire[i] = dwdt(network[0][j].u,
							network[0][j].v, network[0][j].w);
						network[0][j].conc_fire[i] = network[0][j].w;
					}
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
						check+= add_it(&(network[1][j]), j, i, weights_2);
					}
				}

				if(check/ALPHA >= thres)
				{
					network[2][i].activate = 1;
					network[2][i].fire_time = t;
					for (int j = 0; j < hidden_node; ++j)
					{
						network[1][j].deriv_fire[i] = dwdt(network[1][j].u, network[1][j].v, 
							network[1][j].w);
						network[1][j].conc_fire[i] = network[1][j].w;
					}
				}

			}
		}

		update(network, inp_node, hidden_node, output_node);

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
		network[0][i].v = V0;
		network[0][i].u = U0;
		network[0][i].w = W0;
	}
	for (int i = 0; i < hidden; ++i)
	{
		network[1][i].fire_time = max;
		network[1][i].activate = 0;
		network[1][i].v = V0;
		network[1][i].u = U0;
		network[1][i].w = W0;		
	}

	for (int i = 0; i < output; ++i)
	{
		network[2][i].fire_time = max;
		network[2][i].activate = 0;
		network[2][i].v = V0;
		network[2][i].u = U0;
		network[2][i].w = W0;
	}

}


double *back_prop_1(Node **network, double **weights_2, MatrixXd y, 
	int hidden_node, int output_node)
{
	double helpp; 
	double *del = new double[output_node]; 
	double help;
	// printf("k: %i\n", k);
	for (int j = 0; j < output_node; ++j)
	{
		help = 0;

		if (network[2][j].activate)
		{
			for (int i = 0; i < hidden_node; ++i)
			{
				if (network[1][i].activate)
				{
					helpp = weights_2[i][j] * network[1][i].deriv_fire[j]; 
					help += helpp;
				}

			}
		}


		else{help = 1;}
		del[j] = (y(j) - network[2][j].fire_time)/help;
	}
	return del;
}


double *back_prop_2(Node **network, double **weights_1, double **weights_2, double *del,
	int input_node, int hidden_node, int output_node)
{
	double help_num;
	double help_den;
	double *del2 = new double[hidden_node];


	for (int i = 0; i < hidden_node; ++i)
	{

		help_num = 0;
		for (int j = 0; j < output_node; ++j)
		{
			if (network[1][i].activate)
			{
				help_num += del[j] * weights_2[i][j] * network[1][i].deriv_fire[j];
			}
			
		}
		help_den = 0;

//Alternative delta_val: since the derivative of the concentration
// of an input neuron will be zero 0 if that associated hidden node hasn't fired, 
// artificially increase the derivative for an unfired hidden node by making help_den = 1

		if (network[1][i].activate)
		{
			for (int h = 0; h < input_node; ++h)
			{
				help_den += weights_1[h][i] * network[0][h].deriv_fire[i];
			}
		}

		else{help_den = 1;}
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
		printf("  %f,%f,%f   	", network[1][i].fire_time,network[1][i].conc_fire[0], network[1][i].deriv_fire[0]);
	}

		std::cout<<"\n"<<std::endl;

	for (int i = 0; i < output_node; ++i)
	{
		printf("		%f,%f 		", network[2][i].fire_time,network[2][i].w);
	}
	std::cout<<"\n"<<std::endl;

}