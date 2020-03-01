
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
	
	public boolean addLayer(Layer layer) {
		if(layer.size() != outputLayer.getNodes()[0].getNumOuts()) {
			return false;
		}
		this.outputLayer.addNextLayer(layer);
		this.outputLayer = layer;
		return true;
	}
	
	public double[] getOutput(double[] input) {
		Layer currentLayer = this.inputLayer;
		if(currentLayer == null) {
			return null;
		}
		double[] toReturn = input.clone();
		while(currentLayer.getNextLayer() != null) {
			toReturn = currentLayer.getOutput(toReturn);
			currentLayer = currentLayer.getNextLayer();
		}
		return toReturn;
	}
}