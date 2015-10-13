#include "aux_func.h"
using namespace Eigen; 

//max of oregonator: 65.933
// time for spike: 0.409086 'seconds' (of the 10)

int main()
{

	// MatrixXd x = get_data();

	// MatrixXd y = get_y();


	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution_noise(-1,4);
	std::uniform_real_distribution<double> distribution_noise_out(0,4);
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


	const int inp_node = x.cols()+1;
	const int hidden_node = 6;
	const int output_node = y.cols();
	const double dt = 100;
	const int max = MAX;
	// std::ofstream daweights;
	// std::ofstream daweights2;
	std::ofstream outputs;


	Node *inp_layer = new Node[inp_node];
	Node *hidden_layer = new Node[hidden_node];
	Node *output_layer = new Node[output_node];
	Node **network = new Node*[3];
	double learning_param = .5;
	double momentum_param = .9;

	network[0] = inp_layer;
	network[1] = hidden_layer;
	network[2] = output_layer;

// init network
	for (int i = 0; i < inp_node; ++i)
	{
		inp_layer[i].fire_time = max;
		inp_layer[i].activate = 0;
		inp_layer[i].u = U0;
		inp_layer[i].w = W0;
		inp_layer[i].v = V0;
		inp_layer[i].deriv_fire = new double[hidden_node];
		inp_layer[i].conc_fire = new double[hidden_node];
	}

	for (int i = 0; i < hidden_node; ++i)
	{
		hidden_layer[i].fire_time = max;
		hidden_layer[i].activate = 0;
		hidden_layer[i].u = U0;
		hidden_layer[i].w = W0;
		hidden_layer[i].v = V0;
		hidden_layer[i].deriv_fire = new double[output_node];
		hidden_layer[i].conc_fire = new double[output_node];

	}

	for (int i = 0; i < output_node; ++i)
	{
		output_layer[i].fire_time = max;
		output_layer[i].activate = 0;
		output_layer[i].u = U0;
		output_layer[i].v = V0;
		output_layer[i].w = W0;
		output_layer[i].deriv_fire = NULL;
		output_layer[i].conc_fire = NULL;
	}
// init network


//initialize weights
	double ***momentum = new double**[2];


	double **weights_1 = new double*[inp_node];
	double **temp1 = new double*[inp_node];
	momentum[0] = new double*[inp_node];

	

	for (int i = 0; i < inp_node; ++i)
	{
		weights_1[i] = new double[hidden_node];
		temp1[i] = new double[hidden_node];
		momentum[0][i] = new double[hidden_node];

		for (int j = 0; j < hidden_node; ++j)
		{
			temp1[i][j] = 0;
			momentum[0][i][j] = 0;
			weights_1[i][j] = distribution_noise(generator);
		}
		printf("\n");
	}
	printf("\n\n");

	double **weights_2 = new double*[hidden_node];
	double **temp2 = new double*[hidden_node];
	momentum[1] = new double*[hidden_node];

	for (int i = 0; i < hidden_node; ++i)
	{
		temp2[i] = new double[output_node];
		weights_2[i] = new double[output_node];
		momentum[1][i] = new double[output_node];

		for (int j = 0; j < output_node; ++j)
		{
			temp2[i][j] = 0;
			momentum[1][i][j] = 0;
			weights_2[i][j] = distribution_noise(generator);
		}
		printf("\n");
	}

// //initialize weights

// //START WEIGHTS LATER:
// 	MatrixXd weights;
// 	weights = weights1();

// 	for (int i = 0; i < inp_node; ++i)
// 	 {
// 	 	for (int j = 0; j < hidden_node; ++j)
// 	 	{
// 	 		weights_1[i][j] = weights(i,j);
// 	 	}
// 	 } 
// 	 weights = weights2();

// 	 for (int i = 0; i < hidden_node; ++i)
// 	 {
// 	 	for (int j = 0; j < output_node; ++j)
// 	 	{
// 	 		weights_2[i][j] = weights(i,j);
// 	 	}
// 	 }


// //STARTED WEIGHTS LATER:



	double *del1;
	double *del2; 
	int step =0;
	int iter = 12001;
	double cost; 
	outputs.open("xor_outputs.txt");
	// daweights.open("daweights.txt");
	// daweights2.open("daweights2.txt");

	while(step < iter)
	{
		cost = 0;
		for (int tr = 0; tr < x.rows(); ++tr)
		{

			cost += feed_one(network, weights_1, weights_2, x.row(tr), 
				inp_node, hidden_node, output_node, y.row(tr), max);


			del1 = back_prop_1(network, weights_2, y.row(tr), 
				hidden_node, output_node);


			del2 = back_prop_2(network, weights_1, weights_2, del1,
			inp_node, hidden_node, output_node);



			if (step== iter-1 || step == 0)
				{print_network(network, inp_node, hidden_node, output_node);
					printf("\n\n");
					if (step == iter-1)
					{
						for (int i = 0; i < output_node; ++i)
						{
							outputs << std::fixed <<  
							std::setprecision(6) << network[2][i].fire_time << " ";
						}
						outputs << "\n";
						
					}
				}

			if (step ==3000 || step == 3100 || step == 3200 || step ==3300 || step == 3400 || step == 3500
			|| step == 3600 || step == 3700 || step == 3800 || step == 3900 )
			{
				printf("step: %d\n", step);
				print_network(network, inp_node, hidden_node, output_node);
			}

			// if(step == 3300)
			// {
			// 	print_weights(weights_1, inp_node, hidden_node);
			// 	printf("\n\n");
			// 	print_weights(weights_2, hidden_node, output_node);
			// }

			for (int i = 0; i < hidden_node; ++i)
			{
				for (int j = 0; j < output_node; ++j)
				{

					if (network[2][j].activate)
					{
						temp2[i][j] += -1*learning_param*
						network[1][i].conc_fire[j]
						*del1[j]/x.rows();	
					}

				}
			}

			for (int h = 0; h < inp_node; ++h)
			{
				for (int i = 0; i < hidden_node; ++i)
				{
					if (network[1][i].activate)
					{
						temp1[h][i] += -1*learning_param
						*network[0][h].conc_fire[i]
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
				weights_2[i][j] += temp2[i][j] + momentum_param*momentum[1][i][j];
				momentum[1][i][j] = temp2[i][j];
				temp2[i][j]= 0;
				
				// daweights2 << std::fixed <<  std::setprecision(6) << weights_2[i][j] << " ";
			}
			
		}

		// daweights2 << "\n";


		for (int i = 0; i < inp_node; ++i)
		{
			for (int j = 0; j < hidden_node; ++j)
			{
				weights_1[i][j] += temp1[i][j] + momentum_param*momentum[0][i][j];
				momentum[0][i][j] = temp1[i][j];
				temp1[i][j] = 0;
				// weights_1[i][j] = weights_1[i][j];
				// daweights << std::fixed <<  std::setprecision(6) << weights_1[i][j] << " ";

			}
			
		}
		// daweights <<  "\n";

			if (step%1000 == 0){//print_network(network, inp_node, hidden_node, output_node);


			// printf("\n\n\n");
			printf("del1:\n");
			for (int i = 0; i < output_node; ++i)
			{
				printf("%f\n", del1[i]);
			}
			printf("del2:\n");
			for (int i = 0; i < hidden_node; ++i)
			{
				printf("%f\n", del2[i]);
			}

			// for (int i = 0; i < hidden_node; ++i)
			// {
			// 	printf("%f\n", weights_2[i][0]);
			// }
			std::cout << "Cost: " << cost/x.rows() << "step: "<< step << std::endl;
			// std::cin.get();
			}

		step++;
	}


	outputs.close();



	for (int i = 0; i < inp_node; ++i)
	{
		for (int j = 0; j < hidden_node; ++j)
		{
			printf("%f, ", weights_1[i][j]);
		}
		printf("\n");
	}

	printf("\n\n\n");
	for (int i = 0; i < hidden_node; ++i)
	{
		for (int j = 0; j < output_node; ++j)
		{
			printf("%f, ", weights_2[i][j]);
		}
		printf("\n");
	}

//free

	for (int i = 0; i < inp_node; ++i)
	{
		delete[] inp_layer[i].deriv_fire;
		delete[] inp_layer[i].conc_fire;
	}

	for (int i = 0; i < hidden_node; ++i)
	{
		delete[] hidden_layer[i].deriv_fire;
		delete[] hidden_layer[i].conc_fire;
	}
	for (int i = 0; i < output_node; ++i)
	{
		delete[] output_layer[i].deriv_fire;
		delete[] output_layer[i].conc_fire;
	}

	delete[] inp_layer;
	delete[] hidden_layer;
	delete[] output_layer;

	for (int i = 0; i < inp_node; ++i)
	{

		delete[] weights_1[i];
	}

	delete[] weights_1;


	for (int i = 0; i < hidden_node; ++i)
	{

		delete[] weights_2[i];
	}

	delete[] weights_2;


}




