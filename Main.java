import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class Main {

	public static void runSimpleCase() throws InputSizeMismatchException, OutputSizeMismatchException, LayerSizeMismatchException, FileNotFoundException, IOException {
		final int LEARNING_CYCLES = 1000000;
		final double LEARNING_RATE = 0.01;
		
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
	

	public static void runMNIST(int LEARNING_CYCLES, double LEARNING_RATE) throws IOException, LayerSizeMismatchException, InputSizeMismatchException, OutputSizeMismatchException, ClassNotFoundException {
		//final int LEARNING_CYCLES = 10000;
		//final double LEARNING_RATE = 0.1;
		final int TRAINING_ROWS = 42000;
		final int IMAGE_SIZE = 784;
		final int OUTPUT_SIZE = 10;
		final int TESTING_ROWS = 0;
		NeuralNetwork network = NeuralNetwork.loadNeuralNetwork("MNIST");
		
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
			label[index] = 1;
			trainingLabel[a] = label;
			for(int b = 1; b < IMAGE_SIZE + 1; b++) {
				//Normalize data.
				values[b - 1] = Double.parseDouble(data[b]) / 255;
			}
			trainingData[a] = values;
		    a++;
		}
		csvReader.close();

		testingData = Arrays.copyOfRange(trainingData, TRAINING_ROWS - (TRAINING_ROWS / 3), TRAINING_ROWS);
		testingLabel = Arrays.copyOfRange(trainingLabel, TRAINING_ROWS - (TRAINING_ROWS / 3), TRAINING_ROWS);
		
		trainingData = Arrays.copyOfRange(trainingData, 0, TRAINING_ROWS - (TRAINING_ROWS/ 3));
		trainingLabel = Arrays.copyOfRange(trainingLabel, 0, TRAINING_ROWS - (TRAINING_ROWS/ 3));
		
		//trainingData = Arrays.copyOfRange(trainingData, 0, 200);
		//trainingLabel = Arrays.copyOfRange(trainingLabel, 0, 200);
		
		int testValue = 96;
		
		//Before
		input = testingData[testValue];
		expectedOutput = testingLabel[testValue];
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
		input = testingData[testValue];
		expectedOutput = testingLabel[testValue];
		predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n", predictedOutput[0], predictedOutput[1],predictedOutput[2], predictedOutput[3], predictedOutput[4], predictedOutput[5],predictedOutput[6], predictedOutput[7],predictedOutput[8], predictedOutput[9]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		
		//Save that network for further training if needed.
		network.saveNeuralNetwork("MNIST");
	}
	
	public static void main(String[] args) throws InputSizeMismatchException, LayerSizeMismatchException, OutputSizeMismatchException, IOException, ClassNotFoundException {
		//runSimpleCase();
		final int LEARNING_CYCLES = 1000;
		final double LEARNING_RATE = 0.1;
		final int TRAINING_ROWS = 42000;
		final int IMAGE_SIZE = 784;
		final int OUTPUT_SIZE = 10;
		final int TESTING_ROWS = 0;
		
		//NeuralNetwork network = new NeuralNetwork(new SigmoidLayer(IMAGE_SIZE, 10));
		//network.addLayer(new SigmoidLayer(IMAGE_SIZE / 2, IMAGE_SIZE / 2 / 2));
		//network.addLayer(new SigmoidLayer(IMAGE_SIZE / 2 / 2, 10));
		//network.saveNeuralNetwork("MNIST");
		
		runMNIST(LEARNING_CYCLES, LEARNING_RATE);
		
		NeuralNetwork network = NeuralNetwork.loadNeuralNetwork("MNIST");
		
		//network = NeuralNetwork.loadNeuralNetwork("MNIST");
		
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
			label[index] = 1;
			trainingLabel[a] = label;
			for(int b = 1; b < IMAGE_SIZE + 1; b++) {
				//Normalize data.
				values[b - 1] = Double.parseDouble(data[b]) / 255;
			}
			trainingData[a] = values;
		    a++;
		}
		csvReader.close();

		testingData = Arrays.copyOfRange(trainingData, TRAINING_ROWS - (TRAINING_ROWS / 3), TRAINING_ROWS);
		testingLabel = Arrays.copyOfRange(trainingLabel, TRAINING_ROWS - (TRAINING_ROWS / 3), TRAINING_ROWS);
		
		trainingData = Arrays.copyOfRange(trainingData, 0, TRAINING_ROWS - (TRAINING_ROWS / 3));
		trainingLabel = Arrays.copyOfRange(trainingLabel, 0, TRAINING_ROWS - (TRAINING_ROWS/ 3));
		
		//trainingData = Arrays.copyOfRange(trainingData, 0, 200);
		//trainingLabel = Arrays.copyOfRange(trainingLabel, 0, 200);
		//testingData = trainingData;
		//testingLabel = trainingLabel;
		System.out.println(trainingData.length);
		System.out.println(testingData.length);
		testingData = trainingData;
		testingLabel= trainingLabel;
		
		double cost = 0;
		int wrong = 0;
		for(int b = 0; b < testingData.length; b++) {
			input = testingData[b];
			expectedOutput = testingLabel[b];
			predictedOutput = network.getOutputVector(input);
			double[] out = new double[10];
			int maxIndex = 0;
			double max = 0;
			for(int i = 0; i < out.length; i++) {
				if(max < predictedOutput[i]) {
					max = predictedOutput[i];
					maxIndex = i;
				}
			}
			out[maxIndex] = 1;
			int maxIndex2 = 0;
			double max2 = 0;
			for(int i = 0; i < out.length; i++) {
				if(max2 < expectedOutput[i]) {
					max2 = expectedOutput[i];
					maxIndex2 = i;
				}
			}
			cost = cost + network.getCost(out, expectedOutput);
			
			if(network.getCost(out, expectedOutput) > 0) {
				//System.out.printf("Test Case: %d\nExpected: %d\nGot: %d\n\n", b, maxIndex, maxIndex2);
				wrong++;
			}
			
		}
		System.out.println("Average Cost: " + cost / testingData.length);
		System.out.println("Correct: " + (testingData.length - wrong) + " / " + testingData.length);
		//Correct: 25837 / 28000
		
		/*
		int testValue = 1856;
		input = trainingData[testValue];
		expectedOutput = trainingLabel[testValue];
		predictedOutput = network.getOutputVector(input);
		System.out.printf("Predicted: [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n", predictedOutput[0], predictedOutput[1],predictedOutput[2], predictedOutput[3], predictedOutput[4], predictedOutput[5],predictedOutput[6], predictedOutput[7],predictedOutput[8], predictedOutput[9]);
		System.out.println("Expected: " + Arrays.toString(expectedOutput));
		System.out.printf("Cost: %f\n\n", network.getCost(predictedOutput, expectedOutput));
		*/
	}
}
