import java.util.Arrays;

public class Main {

	public static void main(String[] args) throws InputSizeMismatchException, LayerSizeMismatchException, OutputSizeMismatchException {
		NeuralNetwork network = new NeuralNetwork(new SigmoidLayer(10, 20));
		network.addLayer(new SigmoidLayer(20, 3));
		network.addLayer(new SigmoidLayer(3, 2));
		double[] input = new double[] {1,1,1,1,1,0,0,0,0,0};
		double[] expectedOutput = new double[] {1, 0};
		double[] predictedOutput = network.getOutputVector(input);
		System.out.println("Predicted: " + Arrays.toString(predictedOutput));
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.println(network.getCost(predictedOutput, expectedOutput) + "\n");
		System.out.println("Training...\n");
		network.fit(
			new double[][] {
				{0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
				{1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
				{1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
				{1, 0, 1, 1, 1, 0, 0, 0, 0, 0},
				{1, 1, 1, 1 ,1 ,1, 0, 0, 0, 0},
			}, 
			new double[][] {
				{0, 1},
				{1, 0},
				{1, 0},
				{1, 0},
				{1, 0},
			}, 
			1000000, 0.1);
		predictedOutput = network.getOutputVector(input);
		System.out.println("Predicted: " + Arrays.toString(predictedOutput));
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.println(network.getCost(predictedOutput, expectedOutput) + "\n");
		
		input = new double[] {1,0,0,0,1,1,1,1,0,1};
		expectedOutput = new double[] {0, 1};
		predictedOutput = network.getOutputVector(input);
		System.out.println("Predicted: " + Arrays.toString(predictedOutput));
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.println(network.getCost(predictedOutput, expectedOutput) + "\n");
		
		input = new double[] {1,1,0,1,1,0,0,0,0,1};
		expectedOutput = new double[] {1, 0};
		predictedOutput = network.getOutputVector(input);
		System.out.println("Predicted: " + Arrays.toString(predictedOutput));
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.println(network.getCost(predictedOutput, expectedOutput) + "\n");
	}
}
