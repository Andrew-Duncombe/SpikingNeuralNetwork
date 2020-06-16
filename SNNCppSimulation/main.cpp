#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib> 
#include <cstdio>
#include <ctime> 
using namespace std;

#define INPUT_SIZE 784
#define INPUT_LAYER_SIZE 784
#define HIDDEN_LAYER_SIZE 1200
#define OUTPUT_LAYER_SIZE 10
#define SIMULATION_STEP_NUM 35


struct Opts  
{  
	 float t_ref;  
	 float threshold;
	 float dt;
	 float duration;
	 float report_every;
	 float max_rate;
} ;

void read_mnist_labels(char *filename, int labels[], int rows)
{
	int class_num = 10;
	ifstream infile( filename );
	for (int ii = 0; ii < rows; ii++) 
	{
		string s;
		if (!getline( infile, s )) break;

		istringstream ss( s );

		for (int jj = 0; jj < class_num; jj++)
		{
		  string s;
		  if (!getline( ss, s, ',' )) break;
		  if (stoi(s) == 1) labels[ii]= jj;
		}
	}
	infile.close();
}

void read_csv(char *filename, float data[], int rows, int cols)
{
	ifstream infile( filename );
	for (int ii = 0; ii < rows; ii++) 
	{
		string s;
		if (!getline( infile, s )) break;

		istringstream ss( s );

		for (int jj = 0; jj < cols; jj++)
		{
		  string s;
		  if (!getline( ss, s, ',' )) break;
		  data[ii*cols+jj]= stof(s);
		}
	}
	infile.close();
}

	

int main ()
{	
	// Set Random Seed
	srand(static_cast<unsigned int>(time(nullptr))); 
	// Number of MNIST Examples to process
	int num_examples = 100;
	// Read MNIST Labels
	int *mnist_labels = new int[num_examples];
	char mnist_labels_filename[] = "test_labels.txt";
	read_mnist_labels(mnist_labels_filename, mnist_labels, num_examples);
	// Read MNIST Data
	float *mnist_data = new float[num_examples * INPUT_SIZE];
	char mnist_data_filename[] = "test_data.txt";
	read_csv(mnist_data_filename, mnist_data, num_examples, INPUT_SIZE);
	
	// Neural Net Parameters
	Opts t_opts;
	t_opts.t_ref        = 0.000;
	t_opts.threshold    =   1.0;
	t_opts.dt           = 0.001;
	t_opts.duration     = t_opts.dt*SIMULATION_STEP_NUM;
	t_opts.report_every = 0.001;
	t_opts.max_rate     =   200;
	int nn_layerNum = 4;
	int nn_layerSize[nn_layerNum] = {INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE};
	float rescale_fac = 1/(t_opts.dt*t_opts.max_rate);
	
	// Get Neural Net Weight data
	char weights_1[] = "weights_layer1.txt";
	char weights_2[] = "weights_layer2.txt";
	char weights_3[] = "weights_layer3.txt";
	float *layer_hidden1_weights = new float[HIDDEN_LAYER_SIZE * INPUT_LAYER_SIZE];
	float *layer_hidden2_weights = new float[HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE];
	float *layer_output_weights = new float[OUTPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE];
	read_csv(weights_1, layer_hidden1_weights, HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE);
	read_csv(weights_2, layer_hidden2_weights, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE);
	read_csv(weights_3, layer_output_weights, OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
	
	float acc = 0;
	// Run each image through the neural net
	for (int ex = 0; ex < num_examples; ex++) 
	{
		// Run Simulation 
		//if (ex % 50 == 0)
		//{
			printf("Image: %d\n", ex);
		//}
		// Initialize Neuron Layer information
		// Spike Data
		bool *layer_input_spikes = new bool[SIMULATION_STEP_NUM * INPUT_LAYER_SIZE];
		bool *layer_hidden1_spikes = new bool[SIMULATION_STEP_NUM * HIDDEN_LAYER_SIZE];
		bool *layer_hidden2_spikes = new bool[SIMULATION_STEP_NUM * HIDDEN_LAYER_SIZE];
		bool *layer_output_spikes = new bool[SIMULATION_STEP_NUM * OUTPUT_LAYER_SIZE];
		// Membrane Potential
		float *layer_hidden1_mem = new float[SIMULATION_STEP_NUM * HIDDEN_LAYER_SIZE];
		float *layer_hidden2_mem = new float[SIMULATION_STEP_NUM * HIDDEN_LAYER_SIZE];
		float *layer_output_mem = new float[SIMULATION_STEP_NUM * OUTPUT_LAYER_SIZE];
		
		int *layer_output_sum_spikes = new int[OUTPUT_LAYER_SIZE];
		// Generate Image Spike Pattern
		for (int t = 0; t < SIMULATION_STEP_NUM; t++)
		{ 
			for (int p = 0; p < INPUT_SIZE; p++)
			{
				float spike_snapshot = ((float) rand() / (RAND_MAX)) * rescale_fac;
				layer_input_spikes[t*INPUT_LAYER_SIZE + p] = spike_snapshot <= mnist_data[ex*INPUT_SIZE + p];
			}
		}
		
		for (int t = 1; t < SIMULATION_STEP_NUM; t++)
		{
			// Remaining layers
			// Hidden Layer 1
			for (int neuron = 0; neuron < HIDDEN_LAYER_SIZE; neuron++)
			{
				// Calculate neuron impulse by summing weights from spiking synapses
				float impulse = 0;
				for (int synapse = 0; synapse < INPUT_LAYER_SIZE; synapse++)
				{
					if (layer_input_spikes[(t-1)*HIDDEN_LAYER_SIZE + synapse])
					{
						impulse += layer_hidden1_weights[synapse*INPUT_LAYER_SIZE + neuron];
					}
				}
				// Sum impulse to membrane potential
				layer_hidden1_mem[t*HIDDEN_LAYER_SIZE + neuron] = layer_hidden1_mem[(t-1)*HIDDEN_LAYER_SIZE + neuron] + impulse;
				// Check if potential crosses threshold
				if (layer_hidden1_mem[t*HIDDEN_LAYER_SIZE + neuron] >= t_opts.threshold)
				{
					layer_hidden1_spikes[t*HIDDEN_LAYER_SIZE + neuron] = true;
					layer_hidden1_mem[t*HIDDEN_LAYER_SIZE + neuron] = 0;
				}
			}
			// Hidden Layer 2
			for (int neuron = 0; neuron < HIDDEN_LAYER_SIZE; neuron++)
			{
				// Calculate neuron impulse by summing weights from spiking synapses
				float impulse = 0;
				for (int synapse = 0; synapse < HIDDEN_LAYER_SIZE; synapse++)
				{
					if (layer_hidden1_spikes[(t-1)*HIDDEN_LAYER_SIZE + synapse])
					{
						impulse += layer_hidden2_weights[synapse*HIDDEN_LAYER_SIZE + neuron];
					}
				}
				// Sum impulse to membrane potential
				layer_hidden2_mem[t*HIDDEN_LAYER_SIZE + neuron] = layer_hidden2_mem[(t-1)*HIDDEN_LAYER_SIZE + neuron] + impulse;
				// Check if potential crosses threshold
				if (layer_hidden2_mem[t*HIDDEN_LAYER_SIZE + neuron] >= t_opts.threshold)
				{
					layer_hidden2_spikes[t*HIDDEN_LAYER_SIZE + neuron] = true;
					layer_hidden2_mem[t*HIDDEN_LAYER_SIZE + neuron] = 0;
				}
			}
			// Output Layer
			for (int neuron = 0; neuron < OUTPUT_LAYER_SIZE; neuron++)
			{
				// Calculate neuron impulse by summing weights from spiking synapses
				float impulse = 0;
				for (int synapse = 0; synapse < HIDDEN_LAYER_SIZE; synapse++)
				{
					if (layer_hidden2_spikes[(t-1)*OUTPUT_LAYER_SIZE + synapse])
					{
						impulse += layer_output_weights[synapse*HIDDEN_LAYER_SIZE + neuron];
					}
				}
				// Sum impulse to membrane potential
				layer_output_mem[t*OUTPUT_LAYER_SIZE + neuron] = layer_output_mem[(t-1)*OUTPUT_LAYER_SIZE + neuron] + impulse;
				// Check if potential crosses threshold
				if (layer_output_mem[t*OUTPUT_LAYER_SIZE + neuron] >= t_opts.threshold)
				{
					layer_output_spikes[t*OUTPUT_LAYER_SIZE + neuron] = true;
					layer_output_mem[t*OUTPUT_LAYER_SIZE + neuron] = 0;
					layer_output_sum_spikes[neuron]++;
				}
			}
		}
		
		// Determine spiking answer
		int max_class = 0;
		int max_sum_spike = 0;
		for (int spike_class = 0; spike_class < OUTPUT_LAYER_SIZE; spike_class++)
		{
			if (layer_output_sum_spikes[spike_class] >= max_sum_spike)
			{
				max_class = spike_class;
				max_sum_spike = layer_output_sum_spikes[spike_class];
			}
		}
		// Compare with actual answer
		if (max_class == mnist_labels[ex])
		{
			acc++;
		}
		
		// Cleanup SNN data matrices
		delete[] layer_input_spikes;
		layer_input_spikes = nullptr;
		delete[] layer_hidden1_spikes;
		layer_hidden1_spikes = nullptr;
		delete[] layer_hidden2_spikes;
		layer_hidden2_spikes = nullptr;
		delete[] layer_output_spikes;
		layer_output_spikes = nullptr;
		delete[] layer_hidden1_mem;
		layer_hidden1_mem = nullptr;
		delete[] layer_hidden2_mem;
		layer_hidden2_mem = nullptr;
		delete[] layer_output_mem;
		layer_output_mem = nullptr;
		delete[] layer_output_sum_spikes;
		layer_output_sum_spikes = nullptr;
	}
	
	printf("Images Processed End\n");
	// Determine Accuracy
	float snn_accuracy = acc/num_examples * 100;
	printf("Accuracy: %f", snn_accuracy);
	
	return 0;
}