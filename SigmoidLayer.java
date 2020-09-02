
public class SigmoidLayer extends Layer {

	public SigmoidLayer(int inputSize, int outputSize) {
		super(inputSize, outputSize);
	}

	private double sigmoidFunction(double x) {
		return (Math.exp(x) / (Math.exp(x) + 1));
	}
	
	private double derivedSigmoidFunction(double x) {
		return (Math.exp(x) / Math.pow((Math.exp(x) + 1), 2));
	}
	
	@Override
	public double[] applyNonLinearFunction(double[] rawOutputVector) throws InputSizeMismatchException {
		if(rawOutputVector.length != this.outputSize) {
			throw new InputSizeMismatchException("Input size give is not equal to the expected output size.");
		}
		//Create a vector for the output.
		double[] out = new double[this.outputSize];
		for(int a = 0; a < this.outputSize; a++) {
			out[a] = sigmoidFunction(rawOutputVector[a]);
		}
		return out;
	}

	@Override
	public double dCostByDRaw(double expectedValue, double rawValue) {
		return 2 * (sigmoidFunction(rawValue) - expectedValue) * derivedSigmoidFunction(rawValue);
	}

	@Override
	public double applyNonLinearFunction(double rawOutputValue) {
		return sigmoidFunction(rawOutputValue);
	}

	@Override
	public double applyDerivedNonLinearFunction(double rawOutputValue) {
		return derivedSigmoidFunction(rawOutputValue);
	}

}
