import java.util.*;

public class Main {
	
	public static void println(Object a) {
		System.out.println(a);
	}
	
	
	
	public static void main(String[] args) throws LayerSizeMismatchException, UnsupportedMethodException {
		SigmoidLayer test = new SigmoidLayer(5, 2);
		double[] input = {1, 2, 3, 4, 5};
		double[] out = test.getRawOutput(input);
		
		SigmoidLayer test2 = new SigmoidLayer(2, 2);
		
		NeuralNetwork network = new NeuralNetwork(test);
		network.addLayer(test2);
		
		println(Arrays.toString(network.getOutput(input)));
		double[][] allOutputs = network.getAllOutputs(input);
		println(Arrays.deepToString(allOutputs));
		
		double[] raw = test2.getRawOutput(new double[]{0.9999996940977731, 0.9999996940977731});
		double newWeight = test2.getNewWeight(0.8807970137423242, 1, test2.getWeight(0, 0), 0.01, raw[0]);
		println(newWeight);
		network.fit(input, new double[] {1,0}, 100000, 0.1);
		println(Arrays.toString(network.getOutput(input)));
	}
}
