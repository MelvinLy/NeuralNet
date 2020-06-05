
public class ConvolutionalLayer extends Layer {

	//Some methods may not be compatible with this type of layer.
	
	int width;
	int height;
	double[] layer;
	
	//Width and height are width and height of the image.
	public ConvolutionalLayer(int size, int outputSize, int width, int height) {
		super(size, outputSize);
		this.width = width;
		this.height = height;
	}

	public double dCostbyDWeight(int node, int edge, double[] input, double[] expected) throws NullNodeException {

		return 0;
	}

	public double[] activate(double[] input) {
		// TODO Auto-generated method stub
		return null;
	}

}
