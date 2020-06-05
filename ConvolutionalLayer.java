
public class ConvolutionalLayer extends Layer {

	//Some methods may not be compatible with this type of layer.
	
	public ConvolutionalLayer(int size, int outputSize) {
		super(size, outputSize);
		// TODO Auto-generated constructor stub
	}

	public double dCostbyDWeight(int node, int edge, double[] input, double[] expected) throws NullNodeException {

		return 0;
	}

	public double[] activate(double[] input) {
		// TODO Auto-generated method stub
		return null;
	}

}
