import java.util.*;

public class Main {
	
	public static void println(Object a) {
		System.out.println(a);
	}
	
	
	
	public static void main(String[] args) throws LayerSizeMismatchException, UnsupportedMethodException {
		Layer test = new Layer(10, 5);
		NeuralNetwork network = new NeuralNetwork(test);
		for(int a = 0; a < 5; a++) {
			Layer tmp = new SigmoidLayer(5, 5);
			network.addLayer(tmp);
		}
		Layer tmp = new SigmoidLayer(5, 3);
		network.addLayer(tmp);

		double[][] inputs = new double[][] {new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};
		double[][] outputs = new double[][] {new double[] {50, 0, 0}, {0, 0, 0}};
		//network.fit(new double[] {1, 3, 4, 0, 0}, new double[] {1, 0, 0}, 100000, 0.001);
		//network.fit(new double[] {0, 0, 0, 1, 1}, new double[] {0, 0, 1}, 100000, 0.001);
		network.fit(inputs, outputs, 100000, 0.01);
		println(Arrays.toString(network.getOutput(new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})));
		//System.out.println(Arrays.deepToString(network.firstLayer.nextLayer.nextLayer.weights));
		println(Arrays.toString(network.getOutput(new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0})));
		
	}
}
