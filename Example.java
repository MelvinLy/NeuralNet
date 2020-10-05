import java.io.IOException;
import java.util.Arrays;

public class Example {
	//This is an example on how to use the program.
	private static void testTenToTwoNetwork(NeuralNetwork network) throws InputSizeMismatchException, NoLayersException {
		double[] input = new double[] {1,1,1,1,1,0,0,0,0,0};
		double[] expectedOutput = new double[] {1, 0};
		testSingleTenToTwoNetwork(network, input, expectedOutput);

		input = new double[] {1,0,0,0,0,1,1,1,0,1};
		expectedOutput = new double[] {0, 1};
		testSingleTenToTwoNetwork(network, input, expectedOutput);
		
		input = new double[] {1,1,0,1,1,0,0,0,0,1};
		expectedOutput = new double[] {1, 0};
		testSingleTenToTwoNetwork(network, input, expectedOutput);
		
		input = new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		expectedOutput = new double[] {0, 0};
		testSingleTenToTwoNetwork(network, input, expectedOutput);
		
		input = new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
		expectedOutput = new double[] {1, 1};
		testSingleTenToTwoNetwork(network, input, expectedOutput);
	}
	
	private static void testSingleTenToTwoNetwork(NeuralNetwork network, double[] input, double[] expectedOutput) throws InputSizeMismatchException, NoLayersException {
		double[] predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f]\n", predictedOutput[0], predictedOutput[1]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
	}
	
	public static void runSimpleCase() throws InputSizeMismatchException, OutputSizeMismatchException, LayerSizeMismatchException, IOException, NoLayersException, LayerDoesNotExistException {
		final int LEARNING_CYCLES = 100000;
		final double LEARNING_RATE = 0.1;
		
		NeuralNetwork network = new NeuralNetwork(new SigmoidLayer(10, 5));
		network.addLayer(new SigmoidLayer(5, 3));
		network.addLayer(new SigmoidLayer(3, 2));
		System.out.println("--------------------------");
		System.out.println("Initial Output");
		System.out.println("--------------------------\n");
		testTenToTwoNetwork(network);
		
		System.out.println("--------------------------");
		System.out.println("Training...");
		System.out.println("--------------------------\n");
		long start = System.currentTimeMillis();
		double[][] inputs = new double[][] {
			{0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
			{1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
			{1, 0, 1, 1, 1, 0, 0, 0, 0, 0},
			{1, 1, 1, 1 ,1 ,1, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 1, 1, 1, 0, 1},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		};
		double[][] expectedOutputs = new double[][] {
			{0, 1},
			{1, 0},
			{1, 0},
			{1, 0},
			{1, 0},
			{0, 1},
			{0, 0},
			{1, 1},
		};
		network.fit(inputs, expectedOutputs, LEARNING_CYCLES, LEARNING_RATE);
		System.out.printf("Training took: %.2fs\n\n" , (System.currentTimeMillis() - start) / 1000.0);

		testTenToTwoNetwork(network);
		
		System.out.println("--------------------------");
		System.out.println("Clone testing");
		System.out.println("--------------------------\n");
		NeuralNetwork partial = network.clone();
		testTenToTwoNetwork(network);
		partial = network.clone(1, network.getTotalLayers());
		System.out.println("Old Network Size: " + network.getTotalLayers());
		System.out.println("New Network Size: " + partial.getTotalLayers());
		double[] out = partial.getOutputVector(new double[] {1, 1, 1, 1, 1});
		System.out.printf("New Network Output: [%f, %f]\n", out[0], out[1]);
		System.out.println("\n--------------------------");
		System.out.println("Layer Clone Testing");
		System.out.println("--------------------------\n");
		Layer clonedLayer = partial.getLayer(0);
		System.out.println("Cloned Layer Weight Matrix: " + Arrays.deepToString(clonedLayer.weightMatrix));
		
		System.out.println("\n--------------------------");
		System.out.println("Get Gradient Test");
		System.out.println("--------------------------\n");
		network = new NeuralNetwork(new SigmoidLayer(10, 5));
		network.addLayer(new SigmoidLayer(5, 3));
		network.addLayer(new SigmoidLayer(3, 3));
		network.addLayer(new SigmoidLayer(3, 2));
		Object[] gradients = network.getGradients(inputs, expectedOutputs, 0);
		double[][][] weightGradient = (double[][][]) gradients[0];
		double[][] biasGradient = (double[][]) gradients[1];
		System.out.println("Weight Gradients: " + Arrays.deepToString(weightGradient));
		System.out.println("Bias Gradients: " + Arrays.deepToString(biasGradient));
		
		System.out.println("\n--------------------------");
		System.out.println("Manual Fit Test");
		System.out.println("--------------------------\n");
		testTenToTwoNetwork(network);
		System.out.println("--------------------------");
		System.out.println("Training Part 1...");
		System.out.println("--------------------------\n");
		start = System.currentTimeMillis();
		for(int a = 0; a < LEARNING_CYCLES; a++) {
			gradients = network.getGradients(inputs, expectedOutputs, 0);
			weightGradient = (double[][][]) gradients[0];
			biasGradient = (double[][]) gradients[1];
			network.manualFit(weightGradient, biasGradient, LEARNING_RATE, 0, 1);
		}
		System.out.printf("Training took: %.2fs\n\n" , (System.currentTimeMillis() - start) / 1000.0);
		testTenToTwoNetwork(network);
		System.out.println("Weight Gradients: " + Arrays.deepToString((double[][][]) network.getGradients(inputs, expectedOutputs, 0)[0]) + "\n");
		
		System.out.println("--------------------------");
		System.out.println("Training Part 2...");
		System.out.println("--------------------------\n");
		start = System.currentTimeMillis();
		for(int a = 0; a < LEARNING_CYCLES; a++) {
			gradients = network.getGradients(inputs, expectedOutputs, network.getTotalLayers() - 1);
			weightGradient = (double[][][]) gradients[0];
			biasGradient = (double[][]) gradients[1];
			network.manualFit(weightGradient, biasGradient, LEARNING_RATE, network.getTotalLayers() - 1, network.getTotalLayers());
		}
		System.out.printf("Training took: %.2fs\n\n" , (System.currentTimeMillis() - start) / 1000.0);
		testTenToTwoNetwork(network);
		System.out.println("Weight Gradients: " + Arrays.deepToString((double[][][]) network.getGradients(inputs, expectedOutputs, 0)[0]) + "\n");
		
		System.out.println("--------------------------");
		System.out.println("Training Part 3...");
		System.out.println("--------------------------\n");
		start = System.currentTimeMillis();
		for(int a = 0; a < LEARNING_CYCLES; a++) {
			gradients = network.getGradients(inputs, expectedOutputs, 1);
			weightGradient = (double[][][]) gradients[0];
			biasGradient = (double[][]) gradients[1];
			network.manualFit(weightGradient, biasGradient, LEARNING_RATE, 1, network.getTotalLayers() - 1);
		}
		System.out.printf("Training took: %.2fs\n\n" , (System.currentTimeMillis() - start) / 1000.0);
		testTenToTwoNetwork(network);
		System.out.println("Weight Gradients: " + Arrays.deepToString((double[][][]) network.getGradients(inputs, expectedOutputs, 0)[0]));
	}
}
