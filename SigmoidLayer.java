
public class SigmoidLayer extends Layer {
	private static final long serialVersionUID = 1L;

	public SigmoidLayer(int inputSize, int outputSize) {
		super(inputSize, outputSize);
	}

	public SigmoidLayer(int inputSize, int outputSize, int weightFactor) {
		super(inputSize, outputSize, weightFactor);
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
	public double applyNonLinearFunction(double rawOutputValue) {
		return sigmoidFunction(rawOutputValue);
	}

	@Override
	public double applyDerivedNonLinearFunction(double rawOutputValue) {
		return derivedSigmoidFunction(rawOutputValue);
	}

	@Override
	public Layer clone() {
		// TODO Auto-generated method stub
		return null;
	}

}
