import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;

public class Main {

	public static void runSimpleCase() throws InputSizeMismatchException, OutputSizeMismatchException, LayerSizeMismatchException, FileNotFoundException, IOException, NoLayersException {
		final int LEARNING_CYCLES = 1000000;
		final double LEARNING_RATE = 0.1;
		
		NeuralNetwork network = new NeuralNetwork(new SigmoidLayer(10, 5));
		network.addLayer(new SigmoidLayer(5, 3));
		network.addLayer(new SigmoidLayer(3, 2));
		double[] input = new double[] {1,1,1,1,1,0,0,0,0,0};
		double[] expectedOutput = new double[] {1, 0};
		double[] predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f]\n", predictedOutput[0], predictedOutput[1]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		
		input = new double[] {1,0,0,0,0,1,1,1,0,1};
		expectedOutput = new double[] {0, 1};
		predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f]\n", predictedOutput[0], predictedOutput[1]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		
		input = new double[] {1,1,0,1,1,0,0,0,0,1};
		expectedOutput = new double[] {1, 0};
		predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f]\n", predictedOutput[0], predictedOutput[1]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		
		input = new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		expectedOutput = new double[] {0, 0};
		predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f]\n", predictedOutput[0], predictedOutput[1]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		
		input = new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
		expectedOutput = new double[] {1, 1};
		predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f]\n", predictedOutput[0], predictedOutput[1]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		
		System.out.println("--------------------------");
		System.out.println("Training...");
		System.out.println("--------------------------\n");
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
			LEARNING_CYCLES, LEARNING_RATE);
		long end = System.currentTimeMillis();
		
		input = new double[] {1,1,1,1,1,0,0,0,0,0};
		expectedOutput = new double[] {1, 0};
		predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f]\n", predictedOutput[0], predictedOutput[1]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		
		input = new double[] {1,0,0,0,0,1,1,1,0,1};
		expectedOutput = new double[] {0, 1};
		predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f]\n", predictedOutput[0], predictedOutput[1]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		
		input = new double[] {1,1,0,1,1,0,0,0,0,1};
		expectedOutput = new double[] {1, 0};
		predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f]\n", predictedOutput[0], predictedOutput[1]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		
		input = new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		expectedOutput = new double[] {0, 0};
		predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f]\n", predictedOutput[0], predictedOutput[1]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		
		input = new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
		expectedOutput = new double[] {1, 1};
		predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f]\n", predictedOutput[0], predictedOutput[1]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		
		System.out.println("--------------------------");
		System.out.printf("Training took: %.2f s\n", (float) (end - start) / 1000);
		System.out.println("--------------------------");
		
		//network.saveNeuralNetwork("simple.txt");
	}
	
	public static void main(String[] args) throws InputSizeMismatchException, LayerSizeMismatchException, OutputSizeMismatchException, IOException, ClassNotFoundException, NoLayersException {
		runSimpleCase();
	}
}
