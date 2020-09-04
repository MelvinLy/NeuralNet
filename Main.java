import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;

public class Main {

	public static void runSimpleCase() throws InputSizeMismatchException, OutputSizeMismatchException, LayerSizeMismatchException, FileNotFoundException, IOException {
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
		
		network.saveNeuralNetwork("simple.txt");
	}
	
	public static void runMNIST() throws IOException, LayerSizeMismatchException, InputSizeMismatchException, OutputSizeMismatchException {
		final int LEARNING_CYCLES = 10000;
		final double LEARNING_RATE = 0.1;
		final int TRAINING_ROWS = 42000;
		final int IMAGE_SIZE = 784;
		final int OUTPUT_SIZE = 10;
		final int TESTING_ROWS = 0;
		
		double[] input = null;
		double[] expectedOutput = null;
		double[] predictedOutput = null;
		
		double[][] trainingData = new double[TRAINING_ROWS][IMAGE_SIZE];
		double[][] trainingLabel = new double[TRAINING_ROWS][];
		double[][] testingData = new double[TESTING_ROWS][IMAGE_SIZE];
		double[][] testingLabel = new double[TESTING_ROWS][];
		String row = "";
		BufferedReader csvReader = new BufferedReader(new FileReader("train.csv"));
		int a = 0;
		row = csvReader.readLine();
		while ((row = csvReader.readLine()) != null) {
		    String[] data = row.split(",");
			double[] label = new double[OUTPUT_SIZE];
			double[] values = new double[IMAGE_SIZE];
			int index = Integer.parseInt(data[0]);
			label[index] = (double) index;
			trainingLabel[a] = label;
			for(int b = 1; b < IMAGE_SIZE + 1; b++) {
				values[b - 1] = Double.parseDouble(data[b]);
			}
			trainingData[a] = values;
		    a++;
		}
		csvReader.close();

		testingData = Arrays.copyOfRange(trainingData, TRAINING_ROWS - (TRAINING_ROWS / 3), TRAINING_ROWS);
		testingLabel = Arrays.copyOfRange(trainingLabel, TRAINING_ROWS - (TRAINING_ROWS / 3), TRAINING_ROWS);
		
		//trainingData = Arrays.copyOfRange(trainingData, 0, TRAINING_ROWS - (TRAINING_ROWS/ 3));
		//trainingLabel = Arrays.copyOfRange(trainingLabel, 0, TRAINING_ROWS - (TRAINING_ROWS/ 3));
		
		trainingData = Arrays.copyOfRange(trainingData, 0, 200);
		trainingLabel = Arrays.copyOfRange(trainingLabel, 0, 200);
		
		NeuralNetwork network = new NeuralNetwork(new SigmoidLayer(IMAGE_SIZE, 500));
		network.addLayer(new SigmoidLayer(500, 100));
		network.addLayer(new SigmoidLayer(100, 10));
		
		//Before
		input = testingData[0];
		expectedOutput = testingLabel[0];
		predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n", predictedOutput[0], predictedOutput[1],predictedOutput[2], predictedOutput[3], predictedOutput[4], predictedOutput[5],predictedOutput[6], predictedOutput[7],predictedOutput[8], predictedOutput[9]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		//-----------------
		
		System.out.println("--------------------------");
		System.out.println("Training...");
		System.out.println("--------------------------\n");
		long start = System.currentTimeMillis();
		network.fit(trainingData, trainingLabel, LEARNING_CYCLES, LEARNING_RATE);
		long end = System.currentTimeMillis();
		System.out.println("--------------------------");
		System.out.printf("Training took: %.2f s\n", (float) (end - start) / 1000);
		System.out.println("--------------------------");
		
		//After
		input = testingData[0];
		expectedOutput = testingLabel[0];
		predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n", predictedOutput[0], predictedOutput[1],predictedOutput[2], predictedOutput[3], predictedOutput[4], predictedOutput[5],predictedOutput[6], predictedOutput[7],predictedOutput[8], predictedOutput[9]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		
	}
	
	public static void main(String[] args) throws InputSizeMismatchException, LayerSizeMismatchException, OutputSizeMismatchException, IOException, ClassNotFoundException {
		runSimpleCase();
		/*
		double[] input = null;
		double[] expectedOutput = null;
		double[] predictedOutput = null;
		NeuralNetwork network = NeuralNetwork.loadNeuralNetwork("simple.txt");
		input = new double[] {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
		expectedOutput = new double[] {0, 1};
		predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f]\n", predictedOutput[0], predictedOutput[1]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		*/
	}
}
