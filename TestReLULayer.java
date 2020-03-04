public class TestReLULayer extends ReLULayer {

	TestReLULayer() {
		super();
	}
	
	public double[] modifies(double[] in) {
		for(int a = 0; a < in.length; a++) {
			in[a] = in[a];
		}
		return in;
	}
}