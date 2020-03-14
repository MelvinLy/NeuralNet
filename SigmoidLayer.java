public class SigmoidLayer extends Layer {

	public SigmoidLayer(int size, int outputSize) {
		super(size, outputSize);
	}
	
	//Take the input before activation.
	public double[] activate(double[] input) {
		double[] toReturn = new double[input.length];
		for(int a = 0; a < input.length; a++) {
			toReturn[a] = sigmoid(input[a]);
		}
		return toReturn;
	}
	
	//Take input before activation.
	//Edge defines the output node also.
	public double dCostbyDWeight(int node, int edge, double[] input, double[] expected) throws NullNodeException {
		double toReturn = 0;
		double[] outBeforeAct = this.getOutputBeforeAct(input);
		double[] out = this.activate(outBeforeAct);
		for(int a = 0; a < out.length; a++) {
				toReturn = toReturn - (expected[a] - out[a]) * Math.pow(Math.E, outBeforeAct[edge]) * input[node] / Math.pow(Math.pow(Math.E, outBeforeAct[edge]) + 1, 2);
		}
		return toReturn;
	}

	//Squash function.
	public double sigmoid(double x) {
		return Math.pow(Math.E, x) / (Math.pow(Math.E, x) + 1);
	}
}
