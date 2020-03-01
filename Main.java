import java.util.*;

public class Main {
	public static void main(String[] args) throws LayerMismatchException, EmptyLayerException, NodeSizeMismatchException {
		NeuralNet test = new NeuralNet();

		double[] in = {0.1, 1, 0.2, 0.45, -1};
		
		Layer input = NeuralNet.createSigmoidLayer();
		int layerSize = 5;
		int outputSize = 7;
		for(int a = 0; a < layerSize; a++) {
			Node node = Layer.createNode(outputSize);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, 1);
			}
			input.addNode(node);
		}
		test.addLayer(input);
		/////////////////////////////////////////////////////////////////
		Layer layer = NeuralNet.createSigmoidLayer();
		layerSize = 7;
		outputSize = 2;
		for(int a = 0; a < layerSize; a++) {
			Node node = Layer.createNode(outputSize);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, 1);
			}
			layer.addNode(node);
		}
		test.addLayer(layer);
		/////////////////////////////////////////////////////////////////
		layer = NeuralNet.createSigmoidLayer();
		layerSize = 2;
		outputSize = 10;
		for(int a = 0; a < layerSize; a++) {
			Node node = Layer.createNode(outputSize);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, 1);
			}
			layer.addNode(node);
		}
		test.addLayer(layer);
		/////////////////////////////////////////////////////////////////
		layer = NeuralNet.createSigmoidLayer();
		layerSize = 10;
		outputSize = 5;
		for(int a = 0; a < layerSize; a++) {
			Node node = Layer.createNode(outputSize);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, 1);
			}
			layer.addNode(node);
		}
		test.addLayer(layer);
		/////////////////////////////////////////////////////////////////
		layer = NeuralNet.createSigmoidLayer();
		layerSize = 5;
		outputSize = 0;
		for(int a = 0; a < layerSize; a++) {
			Node node = Layer.createNode(outputSize);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, 1);
			}
			layer.addNode(node);
		}
		test.addLayer(layer);
		
		double[] outt = {0.9998477627142086, 0.9998477627142086, 0.9998477627142086, 0.9998477627142086, 0.9998477627142086};
		
		double[] expect = {65, 4322, 6346, 41, 5234};
		//double[] expect = outt;
		
		double idk = layer.dCostByDWeight(0, 0, expect, outt);

		System.out.println(idk);
		
		double[] out = test.getOutput(in);		
		System.out.println(Arrays.toString(out));
		System.out.println(test.size());
		System.out.println(Layer.sigmoidPrime(0.75136507));
		System.out.println("Done.");
	}
}
