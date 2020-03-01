import java.util.*;

public class Main {
	public static void main(String[] args) throws LayerMismatchException, EmptyLayerException {
		NeuralNet test = new NeuralNet();
		
		double[] in = {0.1, 1, 0.2, 0.45, -1};
		
		Layer input = NeuralNet.createLayer();
		for(int a = 0; a < 5; a++) {
			Node node = Layer.createNode(7);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, 0.5);
			}
			input.addNode(node);
		}
		test.addLayer(input);
/////////////////////////////////////////////////////////////////
		Layer layer = NeuralNet.createLayer();
		for(int a = 0; a < 7; a++) {
			Node node = Layer.createNode(7);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, 0.5);
			}
			layer.addNode(node);
		}
		test.addLayer(layer);
/////////////////////////////////////////////////////////////////		
		layer = NeuralNet.createLayer();
		for(int a = 0; a < 7; a++) {
			Node node = Layer.createNode(7);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, 0.5);
			}
			layer.addNode(node);
		}
		test.addLayer(layer);
/////////////////////////////////////////////////////////////////		
		layer = NeuralNet.createLayer();
		for(int a = 0; a < 7; a++) {
			Node node = Layer.createNode(7);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, 0.5);
			}
			layer.addNode(node);
		}
		test.addLayer(layer);
/////////////////////////////////////////////////////////////////		
		double[] out = test.getOutput(in);
		
		System.out.println(Arrays.toString(out));
		System.out.println(test.size());
		System.out.println("Done.");
	}
}
