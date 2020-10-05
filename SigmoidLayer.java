
public class SigmoidLayer extends Layer {
	private static final long serialVersionUID = 1L;

	public SigmoidLayer(int inputSize, int outputSize) {
		super(inputSize, outputSize);
	}

	public SigmoidLayer(int inputSize, int outputSize, double weightFactor) {
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
		if(rawOutputVector.length != this.getOutputSize()) {
			throw new InputSizeMismatchException("Input size give is not equal to the expected output size.");
		}
		//Create a vector for the output.
		double[] out = new double[this.getOutputSize()];
		for(int a = 0; a < this.getOutputSize(); a++) {
			out[a] = sigmoidFunction(rawOutputVector[a]);
		}
		return out;
	}
	
	@Override
	public Layer clone() {
		//Create new layer.
		Layer out = new SigmoidLayer(this.getInputSize(), this.getOutputSize());
		//Copy the weight matrix.
		out.weightMatrix = this.weightMatrix.clone();
		//Copy the bias matrix.
		out.biases = this.biases.clone();
		//Return out.
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

}
