import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class Main {
	public static void main(String[] args) throws LayerMismatchException, EmptyLayerException, NodeSizeMismatchException {
		
		NeuralNet test = new NeuralNet();
		
		int min = 0;
		int max = 1;
		
		Layer input = new TestSigmoidLayer();
		int layerSize = 5;
		int outputSize = 7;
		for(int a = 0; a < layerSize; a++) {
			Node node = Layer.createNode(outputSize);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, ThreadLocalRandom.current().nextDouble(min, max));
			}
			input.addNode(node);
		}
		test.addLayer(input);
		/////////////////////////////////////////////////////////////////
		Layer layer = new TestSigmoidLayer();
		layerSize = 7;
		outputSize = 2;
		for(int a = 0; a < layerSize; a++) {
			Node node = Layer.createNode(outputSize);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, ThreadLocalRandom.current().nextDouble(min, max));
			}
			layer.addNode(node);
		}
		test.addLayer(layer);
		/////////////////////////////////////////////////////////////////
		layer = new TestSigmoidLayer();
		layerSize = 2;
		outputSize = 10;
		for(int a = 0; a < layerSize; a++) {
			Node node = Layer.createNode(outputSize);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, ThreadLocalRandom.current().nextDouble(min, max));
			}
			layer.addNode(node);
		}
		test.addLayer(layer);
		/////////////////////////////////////////////////////////////////
		layer = new TestSigmoidLayer();
		layerSize = 10;
		outputSize = 5;
		for(int a = 0; a < layerSize; a++) {
			Node node = Layer.createNode(outputSize);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, ThreadLocalRandom.current().nextDouble(min, max));
			}
			layer.addNode(node);
		}
		test.addLayer(layer);
		/////////////////////////////////////////////////////////////////
		layer = new TestSigmoidLayer();
		layerSize = 5;
		outputSize = 0;
		for(int a = 0; a < layerSize; a++) {
			Node node = Layer.createNode(outputSize);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, ThreadLocalRandom.current().nextDouble(min, max));
			}
			layer.addNode(node);
		}
		test.addLayer(layer);
		
		double[] in = {0.001, 1, 0.02, 0.045, 0};
		double[] expect = {12, 32, 53, 2, 3};
		double[] out = test.getOutput(in);		
		double idk = layer.dCostByDWeight(0, 0, expect, out);
		System.out.println(Arrays.toString(out));
		System.out.println(test.size());
		System.out.println(idk);		 
	}
}
