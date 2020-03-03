
public class NeuralNet {
	//Make sure to squash your inputs!
	private Layer inputLayer;
	public Layer outputLayer;

	public NeuralNet() {
		this.inputLayer = null;
		this.outputLayer = null;
	}
	
	public NeuralNet(Layer inputLayer) {
		this.inputLayer = inputLayer;
		this.outputLayer = inputLayer;
	}
	
	public static double calculateLoss(double expected, double output) {
		return Math.pow(expected - output, 2) / 2;
	}
	
	public static double calculateCost(double[] expected, double[] output) {
		double cost = 0;
		for(int a = 0; a < expected.length; a++) {
			cost = cost + calculateLoss(expected[a], output[a]);
		}
		return cost;
	}
	
	public void addLayer(Layer layer) throws LayerMismatchException, EmptyLayerException {
		if(this.inputLayer == null) {
			this.inputLayer = layer;
			this.outputLayer = layer;
			return;
		}
		if(layer.size() != outputLayer.getNodes()[0].getNumOuts()) {
			throw new LayerMismatchException("Size of output for this layer does not match input of layer to be added.");
		}
		this.outputLayer.addNextLayer(layer);
		this.outputLayer = layer;
		return;
	}
	
	public double[] getOutput(double[] input) throws LayerMismatchException {
		Layer currentLayer = this.inputLayer;
		if(currentLayer == null) {
			return null;
		}
		double[] toReturn = input;
		while(currentLayer.getNextLayer() != null) {
			toReturn = currentLayer.getOutput(toReturn);
			currentLayer = currentLayer.getNextLayer();
		}
		return toReturn;
	}
	
	public double[] getOutput(double[] input, int depth) throws LayerMismatchException {
		Layer currentLayer = this.inputLayer;
		if(currentLayer == null) {
			return null;
		}
		double[] toReturn = input;
		int a = 0;
		while(currentLayer.getNextLayer() != null && a < depth) {
			toReturn = currentLayer.getOutput(toReturn);
			currentLayer = currentLayer.getNextLayer();
			a++;
		}
		return toReturn;
	}
	
	//Gets you the amount of layers.
	protected int size() {
		int toReturn = 0;
		Layer current = this.inputLayer;
		while(current != null) {
			current = current.getNextLayer();
			toReturn++;
		}
		return toReturn;
	}
}