
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
		double[] toReturn = new double[0];
		while(currentLayer != null) {
			
		}
	}
}