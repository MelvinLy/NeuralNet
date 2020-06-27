import java.util.*;

public class Main {
	
	public static void println(Object a) {
		System.out.println(a);
	}
	
	public static double zeroOne() {
		double t = Math.random();
		if(t > 0.5) {
			return 1;
		}
		else return 0;
	}
	
	public static void main(String[] args) throws LayerSizeMismatchException, UnsupportedMethodException {
		Layer test = new SigmoidLayer(20, 5);
		NeuralNetwork network = new NeuralNetwork(test);
		
		for(int a = 0; a < 5; a++) {
			Layer tmp = new SigmoidLayer(5, 5);
			network.addLayer(tmp);
		}
		
		network.addLayer(new SigmoidLayer(5, 2));

		double[][] inputs = new double[100][20];
		double[][] outputs = new double[100][2];
		for(int a = 0; a < 50; a++) {
			double[] tmp = new double[20];
			for(int b = 0; b < 10; b++) {
				tmp[b] = zeroOne();
			}
			inputs[a] = tmp;
			outputs[a] = new double[] {1, 0};
		}
		for(int a = 50; a < 100; a++) {
			double[] tmp = new double[20];
			for(int b = 10; b < 20; b++) {
				tmp[b] = zeroOne();
			}
			inputs[a] = tmp;
			outputs[a] = new double[] {0, 1};
		}
	
		List<double[]> a1 = Arrays.asList(inputs);
		List<double[]> a2 = Arrays.asList(outputs);
		Collections.shuffle(a1, new Random(12345));
		Collections.shuffle(a2, new Random(12345));
		for(int a = 0; a < 100; a++) {
			inputs[a] = a1.get(a);
		}
		for(int a = 0; a < 100; a++) {
			outputs[a] = a2.get(a);
		}
		/*
		for(int a = 0; a < 100; a++) {
			println(Arrays.toString(inputs[a]));
			println(Arrays.toString(outputs[a]));
		}
		*/
		//println(inputs.length);
		//println(outputs.length);
		println(Arrays.deepToString(network.firstLayer.weights));
		network.fit(inputs, outputs, 1000, 0.1);
		println(Arrays.deepToString(network.firstLayer.weights));
		println(Arrays.toString(inputs[54]));
		println(Arrays.toString(network.getOutput(inputs[0])));
		//println(Arrays.toString(network.getOutput(inputs[1])));
		
	}
}
