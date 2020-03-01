
public class NeuralNet {
	private Layer inputLayer;

	public NeuralNet() {
		this.inputLayer = null;
	}
	
	public NeuralNet(Layer inputLayer) {
		this.inputLayer = inputLayer;
	}
	
	public double[] getOutput(double[] input) {
		Layer currentLayer = this.inputLayer;
		if(currentLayer == null) {
			return null;
		}
		double[] toReturn = input.clone();
		while(currentLayer.getNextLayer() != null) {
			
		}
		return toReturn;
	}
}