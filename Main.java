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
				tmp[b] = 1;
			}
			inputs[a] = tmp;
			outputs[a] = new double[] {1, 0};
		}
		for(int a = 50; a < 100; a++) {
			double[] tmp = new double[20];
			for(int b = 10; b < 20; b++) {
				tmp[b] = 1;
			}
			inputs[a] = tmp;
			outputs[a] = new double[] {0, 1};
		}
	
		List<double[]> a1 = Arrays.asList(inputs);
		List<double[]> a2 = Arrays.asList(outputs);
		
		int s = 3534;
		int i = 98;
		//println(Arrays.toString(inputs[i]));
		Collections.shuffle(a1, new Random(s));
		Collections.shuffle(a2, new Random(s));
		for(int a = 0; a < 100; a++) {
			inputs[a] = a1.get(a);
		}
		for(int a = 0; a < 100; a++) {
			outputs[a] = a2.get(a);
		}
		
		
		println(Arrays.toString(inputs[i]));
		//println(Arrays.toString(outputs[i]));
		/*
		for(int a = 0; a < 100; a++) {
			println(Arrays.toString(inputs[a]));
			println(Arrays.toString(outputs[a]));
		}
		*/
		//println(inputs.length);
		//println(outputs.length);
		//println(Arrays.deepToString(network.firstLayer.nextLayer.weights));
		network.fit(inputs, outputs, 10000, 0.01);
		//println(Arrays.deepToString(network.firstLayer.nextLayer.weights));
		println(Arrays.toString(network.getOutput(inputs[i])));
		println(Arrays.toString(outputs[i]));
		//for(int a = 0; a < inputs.length; a++) {
		//	println(Arrays.toString(network.getOutput(inputs[a])));
		//}
		//println(Arrays.toString(network.getOutput(inputs[1])));
		
	}
}
