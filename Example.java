import java.io.FileNotFoundException;
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
	
	public static void runSimpleCase() throws InputSizeMismatchException, OutputSizeMismatchException, LayerSizeMismatchException, FileNotFoundException, IOException, NoLayersException {
		final int LEARNING_CYCLES = 1000000;
		final double LEARNING_RATE = 0.1;
		
		NeuralNetwork network = new NeuralNetwork(new SigmoidLayer(10, 5));
		network.addLayer(new SigmoidLayer(5, 3));
		network.addLayer(new SigmoidLayer(3, 2));
		testTenToTwoNetwork(network);
		
		System.out.println("--------------------------");
		System.out.println("Training...");
		System.out.println("--------------------------");
		long start = System.currentTimeMillis();
		network.fit(
			new double[][] {
				{0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
				{1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
				{1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
				{1, 0, 1, 1, 1, 0, 0, 0, 0, 0},
				{1, 1, 1, 1 ,1 ,1, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 1, 1, 1, 0, 1},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			}, 
			new double[][] {
				{0, 1},
				{1, 0},
				{1, 0},
				{1, 0},
				{1, 0},
				{0, 1},
				{0, 0},
				{1, 1},
			}, 
			LEARNING_CYCLES, LEARNING_RATE
		);
		System.out.printf("Training took: %.2fs\n\n" , (System.currentTimeMillis() - start) / 1000.0);

		testTenToTwoNetwork(network);
	}
}
