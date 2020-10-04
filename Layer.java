import java.io.Serializable;
import java.util.Random;

public abstract class Layer implements Serializable {
	private static final long serialVersionUID = 1L;
	protected double[][] weightMatrix;
	protected double[] biases;
	protected int inputSize;
	protected int outputSize;

	//Please note the cost function used is (predicted - expected) ^ 2.

	//Creates a new layer.
	public Layer(int inputSize, int outputSize) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.weightMatrix = new double[this.outputSize][this.inputSize];
		this.biases = new double[this.outputSize];
		Random ran = new Random();
		//Populate weightMatrix with random variables.
		//Loop through the rows.
		for(int a = 0; a < outputSize; a++) {
			//Loop through the columns.
			for(int b = 0; b < inputSize; b++) {
				//Assigning random positive and negative values.
				boolean isNegative = ran.nextBoolean();
				if(isNegative) {
					weightMatrix[a][b] = - Math.random() * (double) Math.sqrt(1.0 / inputSize);
				}
				else {
					weightMatrix[a][b] = Math.random() * Math.sqrt(1.0 / inputSize);
				}
			}
			//Assigning random bias value.
			boolean isNegative = ran.nextBoolean();
			if(isNegative) {
				biases[a] = - Math.random() * (double) Math.sqrt(1.0 / inputSize);
			}
			else {
				biases[a] = Math.random() * (double) Math.sqrt(1.0 / inputSize);
			}
		}
	}

	//Creates a new layer. Try changing the weight factor to avoid the vanishing graident. This is defaulted to (1 / inputSize). Set to 1 if this is the first layer.
	public Layer(int inputSize, int outputSize, double weightFactor) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.weightMatrix = new double[this.outputSize][this.inputSize];
		this.biases = new double[this.outputSize];
		Random ran = new Random();
		//Populate weightMatrix with random variables.
		for(int a = 0; a < outputSize; a++) {
			for(int b = 0; b < inputSize; b++) {
				//Assigning random positive and negative values.
				boolean isNegative = ran.nextBoolean();
				if(isNegative) {
					weightMatrix[a][b] = - Math.random() * weightFactor;
				}
				else {
					weightMatrix[a][b] = Math.random() * weightFactor;
				}
			}
			//Assigning random bias value.
			boolean isNegative = ran.nextBoolean();
			if(isNegative) {
				biases[a] = - Math.random() * weightFactor;
			}
			else {
				biases[a] = Math.random() * weightFactor;
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
			//Comment out bias if the layer should not have biases.
			out[a] = currentProduct + biases[a];
		}
		return out;
	}
	
	//Base case for back propagation. Goes from cost right to raw. dCost/dActivation * dActivation/dRawOutput.
	//Each output value has a different value that will be used for the weight that has an affect on it.
	public double dCostByDRaw(double expectedValue, double rawValue) {
		return 2 * (applyNonLinearFunction(rawValue) - expectedValue) * applyDerivedNonLinearFunction(rawValue);
	}

	//Clone function to copy the layer.
	public abstract Layer clone();
	
	//Applies the non-linear function when given the raw output vector. In other words activate.
	public abstract double[] applyNonLinearFunction(double[] rawOutputVector) throws InputSizeMismatchException;

	//Applies the non-linear function when given the raw output vector. In other words activate.
	public abstract double applyNonLinearFunction(double rawOutputValue);

	//The derivative of the non-linear function.
	public abstract double applyDerivedNonLinearFunction(double rawOutputValue);
}
