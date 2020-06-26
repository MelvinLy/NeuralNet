import java.util.*;

public class Main {
	
	public static void println(Object a) {
		System.out.println(a);
	}
	
	public static void main(String[] args) throws LayerSizeMismatchException, UnsupportedMethodException {
		Layer test = new SigmoidLayer(2, 16);
		NeuralNetwork network = new NeuralNetwork(test);
		for(int a = 0; a < 5; a++) {
			Layer tmp = new SigmoidLayer(16, 16);
			network.addLayer(tmp);
		}
		Layer tmp = new SigmoidLayer(16, 3);
		network.addLayer(tmp);

		double[][] inputs = new double[][] {new double[] {10, 0}, new double[] {0, 10}};
		double[][] outputs = new double[][] {new double[] {1, 0, 0}, {0, 0, 1}};
		//network.fit(new double[] {1, 3, 4, 0, 0}, new double[] {1, 0, 0}, 100000, 0.001);
		//network.fit(new double[] {0, 0, 0, 1, 1}, new double[] {0, 0, 1}, 100000, 0.001);
		//println(Arrays.deepToString(network.firstLayer.weights));
		network.fit(inputs[0], outputs[0], 100000, 0.01);

		//println(Arrays.deepToString(network.firstLayer.weights));
		println(Arrays.toString(network.getOutput(new double[] {10, 0})));
		println(Arrays.toString(network.getOutput(new double[] {0, 10})));
		
	}
}
