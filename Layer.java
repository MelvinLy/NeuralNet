
public class Layer {
	private double bias;
	protected Layer previousLayer;
	protected Layer nextLayer;
	protected NeuralNetwork parentNeuralNetwork;
	protected int outputSize;
	protected int inputSize;
	protected double[][] weights; //A row of weight values are designated as weights used to calculate the next respective output node.
	protected double[][] nextSteps;
	
	public Layer(int inputSize, int outputSize) {
		this.weights = new double[outputSize][inputSize];
		this.nextSteps = new double[outputSize][inputSize];
		for(int a = 0; a < this.weights.length; a++) {
			for(int b = 0; b < this.weights[a].length; b++) {
				this.weights[a][b] = Math.random();
			}
		}
		this.bias = 0;
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.nextLayer = null;
	}
	
	public int getOutputSize() {
		return this.outputSize;
	}
	
	public double getWeight(int outputNode, int weightNumber) {
		return weights[outputNode][weightNumber];
	}
	
	//Chain of order: Get partialDerivative, pass result to stepSize, pass to getNewWeight.
	
	//rawExpectedOutput can be anything for the generic layer
	public double getNewWeight(double predictedValue, double expectedValue, double currentWeight, double learningRate, double rawExpectedOutput) throws UnsupportedMethodException {
		double slope = partialDerivative(predictedValue, expectedValue);
		double step = stepSize(slope, learningRate);
		return currentWeight - step;
	}
	
	public double getNextStep(double predictedValue, double expectedValue, double currentWeight, double learningRate, double rawExpectedOutput) throws UnsupportedMethodException {
		double slope = partialDerivative(predictedValue, expectedValue);
		double step = stepSize(slope, learningRate);
		return step;
	}
	
	public void getPreAverageSteps(double[] predicted, double[] expected, double learningRate, double[] rawExpectedOutputs) throws UnsupportedMethodException {
		for(int a = 0; a < weights.length; a++) {
			for(int b = 0; b < weights[a].length; b++) {
				double nWeight = getNextStep(predicted[a], expected[a], weights[a][b], learningRate, rawExpectedOutputs[a]);
				nextSteps[a][b] = nextSteps[a][b] + nWeight;
			}
		}
	}
	
	public void setNewWeight() {
		for(int a = 0; a < nextSteps.length; a++) {
			for(int b = 0; b < nextSteps[a].length; b++) {
				nextSteps[a][b] = nextSteps[a][b] / this.parentNeuralNetwork.getInputSize();
			}
		}
	}
	
	public double cost(double observed, double predicted) {
		return Math.pow(observed - predicted, 2) / 2;
	}
	
	public double stepSize(double slopeFromDerivative, double learningRate) {
		return slopeFromDerivative * learningRate;
	}
	
	public double getNewWeight(double currentWeight, double stepSize) {
		return currentWeight - stepSize;
	}
	
	public double sumOfSquaredResiduals(double[] observed, double[] predicted) {
		double sum = 0;
		for(int a = 0; a < observed.length; a++) {
			sum = sum + cost(observed[a], predicted[a]);
		}
		return sum;
	}
	
	//Need differentiation of ErrorAtWeight by Weight in question. The derivative is partial therefore some of the other weights are treated as constants.
	//(1/2)(Observed - Output) ^ 2
	//Ouput = activation function
	//Activation function = function that contains the output at the node
	//Output at node is a dot product of weights and the inputs
	
	//Should override
	public double partialDerivative(double predictedValue, double expectedValue) throws UnsupportedMethodException {
		return -(expectedValue - predictedValue) * (predictedValue);
	}

	//Should override
	public double[] getActivatedOutput(double[] input) {
		return getRawOutput(input);
	}
	
	public int getInputSize() {
		return this.inputSize;
	}
	
	public double getBias() {
		return this.bias;
	}
	
	public double[] getRawOutput(double[] input) {
		double[] out = new double[outputSize];
		for(int a = 0; a < weights.length; a++) {
			double innerProduct = 0;
			for(int b = 0; b < weights[a].length; b++) {
				innerProduct = innerProduct + weights[a][b] * input[b];
			}
			out[a] = innerProduct;
		}
		return out;
	}
	
}
