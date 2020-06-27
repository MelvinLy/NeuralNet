
public class SigmoidLayer extends Layer {

	public SigmoidLayer(int inputSize, int outputSize) {
		super(inputSize, outputSize);
	}

	private static double sigmoidDerivative(double x) {
		double numerator = Math.exp(x);
		double denominator = Math.pow((numerator + 1), 2);
		return numerator / denominator;
	}

	public double getNewWeight(double predictedValue, double expectedValue, double currentWeight, double learningRate, double rawExpectedOutput, double prev) {
		double slope = partialDerivative(predictedValue, expectedValue, rawExpectedOutput, prev);
		double step = stepSize(slope, learningRate);
		return currentWeight - step;
	}

	public void getPreAverageSteps(double[] predicted, double[] expected, double learningRate, double[] rawExpectedOutputs, double prev[]) throws UnsupportedMethodException {

		for(int a = 0; a < weights.length; a++) {
			for(int b = 0; b < weights[a].length; b++) {
				double nWeight = getNextStep(predicted[a], expected[a], learningRate, rawExpectedOutputs[a], prev[b]);
				nextSteps[a][b] = nextSteps[a][b] + nWeight;
			}
		}

	}

	public double getNextStep(double predictedValue, double expectedValue, double learningRate, double rawExpectedOutput, double prev) throws UnsupportedMethodException {
		double slope = partialDerivative(predictedValue, expectedValue, rawExpectedOutput, prev);
		double step = stepSize(slope, learningRate);
		return step;
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
	public double partialDerivative(double predictedValue, double expectedValue, double rawExpectedOutput, double prev) {
		return -(expectedValue - predictedValue) * sigmoidDerivative(rawExpectedOutput) * (prev);
	}

	public double[] getActivatedOutput(double[] input) {
		double[] out = getRawOutput(input);
		for(int a = 0; a < out.length; a++) {
			out[a] = sigmoid(out[a]);
		}
		return out;
	}

}
