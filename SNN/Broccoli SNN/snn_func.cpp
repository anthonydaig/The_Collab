#include "snn_func.h"
using namespace Eigen;


void reset_network(Node **network, int nodes_layer[], int num_layers)
{
	for (int i = 0; i < num_layers-1; ++i)
	{
		for (int j = 0; j < nodes_layer[i]; ++j)
		{
			network[i][j].activate = 0;
			network[i][j].fire_time = 0;
			network[i][j].deriv_fire = new double[nodes_layer[i+1]];
			network[i][j].conc_fire = new double[nodes_layer[i+1]];
		}
	}

	for (int j = 0; j < nodes_layer[num_layers-1]; ++j)
	{
		network[num_layers-1][j].activate = 0;
		network[num_layers-1][j].fire_time = 0;
		network[num_layers-1][j].conc_fire = NULL;
		network[num_layers-1][j].deriv_fire = NULL;

	}
}


void print_network(Node **network, int nodes_layer[], int num_layers)
{
	for (int i = 0; i < num_layers; ++i)
	{
		for (int j = 0; j < nodes_layer[i]; ++j)
		{
			printf("yes: %d, fire: %f 		",
			 network[i][j].activate, network[i][j].fire_time);
		}
		printf("\n");
	}

}