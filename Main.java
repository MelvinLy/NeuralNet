import java.util.Arrays;

public class Main {
	
	public static void print(Object a) {
		System.out.println(a);
	}
	
	public static void main(String args[]) throws LayerSizeMismatchException, NullNodeException, NodeSizeMismatchException {
		
		SigmoidLayer layer = new SigmoidLayer(5, 3);
		SigmoidLayer layer2 = new SigmoidLayer(3, 3);
		
		for(int a = 0; a < layer.size(); a++) {
			layer.setNode(a, new Node(3));
		}
		
		for(int a = 0; a < layer2.size(); a++) {
			layer2.setNode(a, new Node(3));
		}
		
		NeuralNetwork net = new NeuralNetwork(layer, layer2);
		
		double[] input = {1,2,3,4,5};
		double[] output = net.getOutput(input);
		double[] expected = {0.9525740853634935, 0.9525740853634935, 0.9525740853634935};
		
		print(Arrays.toString(output) + "\n");
		print(net.getNewWeight(1, 0, 0, input, expected, 0.1));
	}
}
