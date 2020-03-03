import java.util.*;
import java.util.concurrent.ThreadLocalRandom;



public class Main {
	public static void print(Object x) {
		System.out.println(x);
	}
	public static void main(String[] args) throws LayerMismatchException, EmptyLayerException, NodeSizeMismatchException {
		
		NeuralNet test = new NeuralNet();
		
		int min = 0;
		int max = 1;
		
		/*
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
		
		layer = new TestSigmoidLayer();
		layerSize = 5;
		outputSize = 0;
		for(int a = 0; a < layerSize; a++) {
			Node node = Layer.createNode(outputSize);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, 1);
			}
			layer.addNode(node);
		}*/
		//test.addLayer(layer);
		
		SigmoidLayer input = new TestSigmoidLayer();
		int layerSize = 5;
		int outputSize = 2;
		for(int a = 0; a < layerSize; a++) {
			Node node = Layer.createNode(outputSize);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, 1);
			}
			input.addNode(node);
		}
		test.addLayer(input);
		/////////////////////////////////////////////////////////////////
		SigmoidLayer layer = new TestSigmoidLayer();
		layerSize = 2;
		outputSize = 4;
		for(int a = 0; a < layerSize; a++) {
			Node node = Layer.createNode(outputSize);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, 1);
			}
			layer.addNode(node);
		}
		test.addLayer(layer);
		/////////////////////////////////////////////////////////////////
		//Output layer
		layer = new TestSigmoidLayer();
		layerSize = 4;
		outputSize = 0;
		for(int a = 0; a < layerSize; a++) {
			Node node = Layer.createNode(outputSize);
			for(int b = 0; b < node.getNumOuts(); b++) {
				node.setMultiplier(b, 1);
			}
			layer.addNode(node);
		}
		test.addLayer(layer);
		
		double[] in = {1,2,3,4,5};
		double[] out = test.getOutput(in, 3);
		
		//print(Arrays.toString(out));
		
		//print(Arrays.toString(out));
		
		//print(Arrays.toString(layer.parentLayer.getOutput(out)));
		/*
		in = out;
		out = layer.parentLayer.getOutput(out);
		print(Arrays.toString(in));
		print(Arrays.toString(out));
		*/
		SigmoidLayer parent = (SigmoidLayer) layer.parentLayer;
		//print(parent.dCostByDWeightSig(test, 3, 0, 0, out, in));
		double[][] all = test.getAllOutputs(in);
		for(int a = 0; a < all.length; a++) {
			//print(Arrays.toString(all[a]));
		}
		double[] inputs = {0.9933071490757152,0.9933071490757152};
		double[] expected = {1765.11780465735511428, 1.11780465735511428, 1.11780465735511428, 1.1135511428};
		//print(layer.dCostByDWeightSig(test, 2, 0, 0, expected, in));
	}
}
