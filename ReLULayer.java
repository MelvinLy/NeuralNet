
public class ReLULayer extends Layer {

	private static final long serialVersionUID = 1L;

	public ReLULayer(int inputSize, int outputSize) {
		super(inputSize, outputSize);
	}

	public double reLUFunction(double a) {
		if(a > 0) {
			return a;
		}
		return 0;
	}
	
	public double derivedReLUFunction(double a) {
		if(a >= 0) {
			return 1;
		}
		return 0;
	}
	
	@Override
	public double[] applyNonLinearFunction(double[] rawOutputVector) throws InputSizeMismatchException {
		double[] out = new double[rawOutputVector.length];
		for(int a = 0; a < rawOutputVector.length; a++) {
			out[a] = reLUFunction(rawOutputVector[a]);
		}
		return out;
	}

	@Override
	public double applyNonLinearFunction(double rawOutputValue) {
		return reLUFunction(rawOutputValue);
	}

	@Override
	public double dCostByDRaw(double expectedValue, double rawValue) {
		return 2 * (reLUFunction(rawValue) - expectedValue) * derivedReLUFunction(rawValue);
	}

	@Override
	public double applyDerivedNonLinearFunction(double rawOutputValue) {
		return derivedReLUFunction(rawOutputValue);
	}

}
