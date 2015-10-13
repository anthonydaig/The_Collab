#include "snn_func.h"

using namespace Eigen;



int main()
{


//initialize inputs (for XOR)
	MatrixXd x(4,2);
	x<< 0,0,
		0,1,
		1,0,
		1,1;

	//initialize outputs (for XOR)
	MatrixXd y(4,1);
	y<< 2.5,
		2.0,
		2.0,
		2.5;


	const int num_layers = 3;
	int nodes_layer[num_layers];

	// we need an array that holds the number of nodes in
	// each layer

	nodes_layer[0] = x.cols();

	//for now
	nodes_layer[1] = 6;
	//
	nodes_layer[num_layers-1] = y.cols();
	//

	Node **network = new Node*[num_layers];

	for (int i = 0; i < num_layers; ++i)
	{
		network[i] = new Node[nodes_layer[i]];
	}


	reset_network(network, &nodes_layer[0], num_layers);

	print_network(network, &nodes_layer[0], num_layers);




















}



