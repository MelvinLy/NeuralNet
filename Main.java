public class Main {
	public static void main(String[] args) {
		NeuralNet test = new NeuralNet();
		
		Layer input = NeuralNet.createLayer(0.05);
		for(int a = 0; a < 5; a++) {
			Node node = Layer.createNode(7);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, 0.5);
			}
		}
		
	}
}
