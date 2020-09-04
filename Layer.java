import java.io.Serializable;

public abstract class Layer implements Serializable {
	private static final long serialVersionUID = 1L;
	protected double[][] weightMatrix;
	protected int inputSize;
	protected int outputSize;

	//Please note the cost function used is (predicted - expected)^2.

	//Creates a new layer.
	public Layer(int inputSize, int outputSize) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.weightMatrix = new double[this.outputSize][this.inputSize];
		//Populate weightMatrix with random variables.
		for(int a = 0; a < outputSize; a++) {
			for(int b = 0; b < inputSize; b++) {
				weightMatrix[a][b] = Math.random();
			}
		}
	}

	//Creates a new layer. Try increasing the weight factor if the outputs are consistently all zeroes or ones.
	public Layer(int inputSize, int outputSize, int weightFactor) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.weightMatrix = new double[this.outputSize][this.inputSize];
		//Populate weightMatrix with random variables.
		for(int a = 0; a < outputSize; a++) {
			for(int b = 0; b < inputSize; b++) {
				weightMatrix[a][b] = Math.random() / weightFactor;
			}
		}
	}

	//Returns the output vector before applying the non-linear function.
	public double[] getRawOutput(double[] input) throws InputSizeMismatchException {
		if(input.length != this.inputSize) {
			throw new InputSizeMismatchException("Input's size given is not equal to the expected layer input size.");
		}
		//Create an output vector that has the size of the output.
		double[] out = new double[this.outputSize];
		for(int a = 0; a < this.outputSize; a++) {
			//Compute product of each row.
			double[] weightRow = weightMatrix[a];
			double currentProduct = 0;
			for(int b = 0; b < this.inputSize; b++) {
				currentProduct = currentProduct + weightRow[b] * input[b];
			}
			out[a] = currentProduct;
		}
		return out;
	}

	//Applies the non-linear function when given the raw output vector. In other words activate.
	public abstract double[] applyNonLinearFunction(double[] rawOutputVector) throws InputSizeMismatchException;

	//Applies the non-linear function when given the raw output vector. In other words activate.
	public abstract double applyNonLinearFunction(double rawOutputValue);

	//Base case for back propagation. Goes from cost right to raw. dCost/dActivation * dActivation/dRawOutput.
	//Each output value has a different value that will be used for the weight that has an affect on it.
	public abstract double dCostByDRaw(double expectedValue, double rawValue);

	//The derivative of the non-linear function.
	public abstract double applyDerivedNonLinearFunction(double rawOutputValue);
}
