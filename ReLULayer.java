public class ReLULayer extends Layer {

	public ReLULayer(int size, int outputSize) {
		super(size, outputSize);
	}
	
	public double[] activate(double[] input) {
		double[] toReturn = new double[input.length];
		for(int a = 0; a < input.length; a++) {
			toReturn[a] = reLU(input[a]);
		}
		return toReturn;
	}
	
	public double dCostbyDWeight(int node, int edge, double[] input, double[] expected) throws NullNodeException {
		double toReturn = 0;
		double[] outBeforeAct = this.getOutputBeforeAct(input);
		double[] out = this.activate(outBeforeAct);
		for(int a = 0; a < out.length; a++) {
				if(out[a] > 0) {
					toReturn = toReturn + (expected[a] - out[a]) *  input[node];
				}
		}
		return toReturn;
	}
	
	public double reLU(double x) {
		if(x < 0) {
			return 0;
		}
		return x;
	}
}
