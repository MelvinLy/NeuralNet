
public class NeuralNet {
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
	
	public static SigmoidLayer createLayer(double k) {
		return new SigmoidLayer(k);
	}
	
	public static ReLULayer createLayer() {
		return new ReLULayer();
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
	
	public double[] getOutput(double[] input) {
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