// test neural net

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <conio.h>

#include <fstream>
#include <sstream>

using namespace std;

// ********************** class TrainingData ***********************

class TrainingData
{
public:
	TrainingData(const string filename);
	bool isEof(void) { return trainingDataFile.eof(); }
	void getTopology(vector<unsigned> &topology);

	// Returns the number of input values read from the file:
	unsigned getNextInputs(vector<double> &inputVals);
	unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
	ifstream trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
	string line;
	string label;

	getline(trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("topology:") != 0) {
		abort();
	}

	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

	return;
}

TrainingData::TrainingData(const string filename)
{
	trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
	inputVals.clear();

	string line;
	getline(trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
	targetOutputVals.clear();

	string line;
	getline(trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}




struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ********************** class Neuron ***********************

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned index);
	void setOutputValue(double value) { outputValue = value; }
	double getOutputValue() const { return outputValue; }
	void feedForward(const Layer &prevLayer);
	void calculateOutputGradients(double targetValue);
	void calculateHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);

private:
	static double eta; // [0.0..1.0] overall net training rate
	static double alpha; // [0.0..n] multiplier of last weight change (momentum)
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double outputValue;
	vector<Connection> outputWeights;
	unsigned wIndex;
	double gradient;	
};

double Neuron::eta = 0.15; // overall net learning rate
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight. [0.0..n]


void Neuron::updateInputWeights(Layer &prevLayer)
{
	// The weights to be updated are in the Connection container
	// in the neurons in the preceding layer

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.outputWeights[wIndex].deltaWeight;

		double newDeltaWeight =
			// Individual input, magnified by the gradient and train rate:
			eta
			* neuron.getOutputValue()
			* gradient
			// Also add momentum = a fraction of the previous delta weight
			+ alpha
			* oldDeltaWeight;

		neuron.outputWeights[wIndex].deltaWeight = newDeltaWeight;
		neuron.outputWeights[wIndex].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	// Sum our contributions of the error at the nodes we feed

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}

	return sum;
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	gradient = dow * Neuron::transferFunctionDerivative(outputValue);
}

void Neuron::calculateOutputGradients(double targetValue)
{
	double delta = targetValue - outputValue;
	gradient = delta * Neuron::transferFunctionDerivative(outputValue);
}

double Neuron::transferFunction(double x)
{
	// tanh - output range [-1.0..1.0]
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	// tanh derivative
	return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	// Sum the previous layerNum's outputs (which are our inputs)
	// Include the bias node from the previous layerNum.

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		sum += prevLayer[n].getOutputValue() * 
			   prevLayer[n].outputWeights[wIndex].weight;
	}

	outputValue = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned index)
{
	for (unsigned c = 0; c < numOutputs; ++c) {
		outputWeights.push_back(Connection());
		outputWeights.back().weight = randomWeight();		
	}

	wIndex = index;
}

// ********************** class Net ***********************

class Net
{
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<double> &inputValues);
	void backPropagation(const vector<double> &targetValues);
	void results(vector<double> &outputValues) const;
	double getRecentAverageError(void) const { return nRecentAverageError; }
	
private:
	vector<Layer> layers; // layerNum[layerNum][neuron]
	double nError;
	double nRecentAverageError;
	static double nRecentAverageSmoothingFactor;
};


double Net::nRecentAverageSmoothingFactor = 100.0; // Number of training samples to average over


void Net::results(vector<double> &outputValues) const 
{
	outputValues.clear();

	for (unsigned n = 0; n < layers.back().size(); ++n) {
		outputValues.push_back(layers.back()[n].getOutputValue());
	}
}

void Net::backPropagation(const vector<double> &targetValues) 
{
	// Calculate overall net error (RMS of output neuron errors)

	Layer &outputLayer = layers.back();
	nError = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		double delta = targetValues[n] - outputLayer[n].getOutputValue();
		nError += delta * delta;
	}
	nError /= outputLayer.size() - 1; // get average error squared
	nError = sqrt(nError); // RMS

	// Implement a recent average measurement

	nRecentAverageError =
		(nRecentAverageError * nRecentAverageSmoothingFactor + nError) /
		(nRecentAverageSmoothingFactor + 1);

	// Calculate output layerNum gradients

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		outputLayer[n].calculateOutputGradients(targetValues[n]);
	}

	// Calculate gradients on hidden layers

	for (unsigned layerNum = layers.size() - 2; layerNum > 0; --layerNum) {
		Layer &hiddenLayer = layers[layerNum];
		Layer &nextLayer = layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calculateHiddenGradients(nextLayer);
		}
	}

	// For all alyers from outputs to first hidden layerNum.
	// update connection weights

	for (unsigned layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
		Layer &currLayer = layers[layerNum];
		Layer &prevLayer = layers[layerNum - 1];

		for (unsigned n = 0; n < currLayer.size() - 1; ++n) {
			currLayer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::feedForward(const vector<double> &inputValues) 
{
	assert(inputValues.size() == layers[0].size() - 1);

	// Assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputValues.size(); ++i) {
		layers[0][i].setOutputValue(inputValues[i]);
	}

	// Forward propagate
	for (unsigned layerNum = 1; layerNum < layers.size(); ++ layerNum) {
		Layer &prevLayer = layers[layerNum - 1];
		for (unsigned n = 0; n < layers[layerNum].size() - 1; ++n) {
			layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

Net::Net(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// We have a new layer, now fill it with neurons, and
		// add a bias neuron in each layer.
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "Made a Nueron!" << endl;
		}

		// Force the bias node's output value to 1.0. 
		// It's the last neuron created abover
		layers.back().back().setOutputValue(1.0);
	}
}


void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		cout << v[i] << " ";
	}

	cout << endl;
}


void makeTesttrainingDataFile()
{
	// Random training sets for XOR -- two inputs and one output
	ofstream outfile("trainingData.txt");

	outfile << "topology: 2 4 1" << endl;
	for (int i = 2000; i >= 0; --i) {
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		int n2 = (int)(2.0 * rand() / double(RAND_MAX));
		int t = n1 ^ n2; // should be 0 or 1
		outfile << "in: " << n1 << ".0 " << n2 << ".0 " << endl;
		outfile << "out: " << t << ".0" << endl;
	}

	outfile.close();
}

int main()
{
	makeTesttrainingDataFile();
	TrainingData trainData("trainingData.txt");

	// e.g., { 2, 4, 1 }
	vector<unsigned> topology;
	trainData.getTopology(topology);
	
	Net net(topology);
	
	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	
	while (!trainData.isEof()) {
		++trainingPass;
		cout << endl << "Pass " << trainingPass;
	
		// Get new input data and feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0]) {
			break;
		}
		showVectorVals(": Inputs:", inputVals);
		net.feedForward(inputVals);
	
		// Collect the net's actual output results:
		net.results(resultVals);
		showVectorVals("Outputs:", resultVals);
	
		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);

		cout << "asserting" << endl;
		assert(targetVals.size() == topology.back());
	
		net.backPropagation(targetVals);
	
		// Report how well the training is working, average over recent samples:
		cout << "Net recent average error: "
			 << net.getRecentAverageError() << endl;
	}

	cout << endl << "Done" << endl;

	getch();
}