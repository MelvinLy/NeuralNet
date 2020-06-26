
public class SigmoidLayer extends Layer {

	public SigmoidLayer(int inputSize, int outputSize) {
		super(inputSize, outputSize);
	}

	private static double sigmoidDerivative(double x) {
		double numerator = Math.exp(x);
		double denominator = Math.pow((numerator + 1), 2);
		return numerator / denominator;
	}
	
	public double getNewWeight(double predictedValue, double expectedValue, double currentWeight, double learningRate, double rawExpectedOutput) {
		double slope = partialDerivative(predictedValue, expectedValue, rawExpectedOutput);
		double step = stepSize(slope, learningRate);
		return currentWeight - step;
	}
	
	//May need this again
	/*
	public double getNewWeight(double predictedValue, double expectedValue, double currentWeight, double learningRate) throws UnsupportedMethodException {
		throw new UnsupportedMethodException("This method is not available for this layer.");
	}
	*/
	
	/*
	 public double stepSize(double slopeFromDerivative, double learningRate) {
		return slopeFromDerivative * learningRate;
	}
	
	public double getNewWeight(double weight, double stepSize) {
		return weight - stepSize;
	}
	*/
	
	private static double sigmoid(double x) {
		double numerator = Math.exp(x);
		double denominator = numerator + 1;
		return numerator / denominator;
	}
	
	public double partialDerivative(double predictedValue, double expectedValue) throws UnsupportedMethodException {
		throw new UnsupportedMethodException("This method is not available for this layer.");
	}
	
	//Predicted and expected values are value of node after applying the activation function.
	//Raw output is the value before applying the activation function. 
	public double partialDerivative(double predictedValue, double expectedValue, double rawExpectedOutput) {
		return -(expectedValue - predictedValue) * sigmoidDerivative(rawExpectedOutput) * (predictedValue);
	}
	
	public double[] getActivatedOutput(double[] input) {
		double[] out = getRawOutput(input);
		for(int a = 0; a < out.length; a++) {
			out[a] = sigmoid(out[a]);
		}
		return out;
	}
	
}
