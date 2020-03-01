
public class NeuralNet {
	private Layer inputLayer;
	private Layer outputLayer;

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
	
	public void addLayer(Layer layer) {
		if(layer.size() != outputLayer.getNodes()[0].getNumOuts()) {
			this.inputLayer = layer;
		}
		this.outputLayer.addNextLayer(layer);
		this.outputLayer = layer;
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
}