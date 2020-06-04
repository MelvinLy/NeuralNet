import java.util.Arrays;

public class NeuralNetwork {
	private Layer inputLayer;
	private Layer lastLayer;
	private int size;
	
	public NeuralNetwork(Layer inputLayer, Layer lastLayer) throws LayerSizeMismatchException {
		this.inputLayer = inputLayer;
		this.lastLayer = lastLayer;
		this.inputLayer.addNextLayer(lastLayer);
		this.inputLayer.parentNetwork = this;
		this.lastLayer.parentNetwork = this;
		this.inputLayer.layerNumber = 0;
		this.lastLayer.layerNumber = 1;
		this.size = 2;
	}
	
	public void addLayer(Layer layer) throws LayerSizeMismatchException {
		this.lastLayer.addNextLayer(layer);
		this.lastLayer = layer;
		this.lastLayer.parentNetwork = this;
		this.size++;
		this.lastLayer.layerNumber = size - 1;
	}
	
	public double[][] getAllOutputs(double[] input) throws NullNodeException {
		double[][] toReturn = new double[this.size()][];
		double[] carry = input.clone();
		Layer currentLayer = this.inputLayer;
		for(int a = 0; a < this.size(); a++) {
			double[] temp = currentLayer.getOutputBeforeAct(carry);
			temp = currentLayer.activate(temp);
			toReturn[a] = temp;
			carry = temp;
			currentLayer = currentLayer.nextLayer;
		}
		return toReturn;
	}
	
	public double getNewWeight(int layerNumber, int node, int edge, double[] input, double[] expected, double rate) throws NullNodeException {
		Layer currentLayer = this.inputLayer;
		while(currentLayer.layerNumber != layerNumber) {
			currentLayer = currentLayer.nextLayer;
		}
		return currentLayer.getNewWeight(node, edge, input, expected, rate);
	}
	
	public int size() {
		return size;
	}
	
	public double[] getOutput(double[] input) throws NullNodeException {
		double[][] temp = this.getAllOutputs(input);
		return temp[this.size - 1];
	}
}