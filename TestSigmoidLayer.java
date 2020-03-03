public class TestSigmoidLayer extends SigmoidLayer {

	TestSigmoidLayer() {
		super();
	}
	
	public double[] modifies(double[] in) {
		for(int a = 0; a < in.length; a++) {
			in[a] = in[a] - 2;
		}
		return in;
	}
}
