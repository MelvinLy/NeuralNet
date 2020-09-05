import java.util.ArrayList;
import java.util.Arrays;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

public class NeuralNetwork implements Serializable {
	private static final long serialVersionUID = 1L;
	private ArrayList<Layer> allLayers;
	
	//Creates a new network with a single compute layer.
	public NeuralNetwork(Layer layer) {
		this.allLayers = new ArrayList<Layer>();
		//Add the layer to the List of all ordered layers.
		allLayers.add(layer);
	}
	
	//Method to save the network.
	public void saveNeuralNetwork(String fileName) throws IOException {
		//Create the output stream.
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName));
		//Write to file.
		out.writeObject(this);
		//Close stream.
		out.close();
	}
	
	//Method to load to a saved network.
	public static NeuralNetwork loadNeuralNetwork(String fileName) throws FileNotFoundException, IOException, ClassNotFoundException {
		//Create the output stream.
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
		//Read and cast object.
		NeuralNetwork network = (NeuralNetwork) in.readObject();
		//Close stream.
		in.close();
		//Return network.
		return network;
	}
	
	//Add a new compute layer to the network.
	public void addLayer(Layer layer) throws LayerSizeMismatchException {
		//Check if the last layers output is equal to the input size of the given layer.
		if(allLayers.get(allLayers.size() - 1).outputSize != layer.inputSize) {
			throw new LayerSizeMismatchException("The output size of the last layer is not equal to the input size of the layer given.");
		}
		allLayers.add(layer);
	}
	
	public double[] getOutputVector(double[] input) throws InputSizeMismatchException {
		if(input.length != allLayers.get(0).inputSize) {
			throw new InputSizeMismatchException("Input's size given is not equal to the expected layer input size.");
		}
		double[] out = input;
		for(int a = 0; a < allLayers.size(); a++) {
			//Fetch current layer.
			Layer currentLayer = allLayers.get(a);
			//Computer the raw output for the current layer.
			double[] rawOutput = currentLayer.getRawOutput(out);
			//Apply the activation function.
			out = currentLayer.applyNonLinearFunction(rawOutput);
		}
		return out;
	}
	
	//Calculate cost based on predicted output and expected output.
	public double getCost(double[] predictedOutput, double[] expectedOutput) throws InputSizeMismatchException {
		//Checks if both arrays are equal length.
		if(predictedOutput.length != expectedOutput.length) {
			throw new InputSizeMismatchException("The size of the two given arrays are not equal.");
		}
		double out = 0;
		//Compute the total cost between the predicted output and expected output.
		for(int a = 0; a < predictedOutput.length; a++) {
			out = out + Math.pow(predictedOutput[a] - expectedOutput[a], 2);
		}
		return out;
	}
	
	//Create a neural network model.
	public void fit(double[][] inputs, double[][] expectedOutputs, int trainCycles, double learningRate) throws InputSizeMismatchException, OutputSizeMismatchException {
		for(int a = 0; a < trainCycles; a++) {
			//Collection of all weight adjustments averaged. Positive gradient at the moment.
			double[][][] adjustmentMatrices = new double[allLayers.size()][][];
			//Bias adjustment vector.
			double[] biasAdjustment = new double[allLayers.size()];
			//Loop through to create appropriate size of the weight adjustment for each layer.
			//Create the 3D adjustment matrix.
			for(int c = allLayers.size() - 1; c >= 0; c--) {
				//Fetch the current layer.
				Layer currentLayer = allLayers.get(c);
				//Create the gradient adjustment matrix at the current layer.
				adjustmentMatrices[c] = new double[currentLayer.weightMatrix.length][currentLayer.weightMatrix[0].length];
			}
			//Looping through all given inputs as training data.
			for(int b = 0; b < inputs.length; b++) {
				//Check that the corresponding output has the expected network output size.
				if(expectedOutputs[b].length != this.allLayers.get(allLayers.size() - 1).outputSize) {
					throw new OutputSizeMismatchException("The expected output and the network's output are not equal in size.");
				}
				//The current training data.
				double[] currentInput = inputs[b];
				//Create an array of the raw outputs at each layer.
				double[][] allRawOutputs = new double[allLayers.size()][];
				//The previous running product computed at each layer.
				double[] prev = currentInput;
				//The derivative bases that will be used in backpropagation. Stores up to the raw values.
				double[][] derivatives = new double[allLayers.size()][];
				//The derivatives for the bias.
				double[] biasDerivatives = new double[allLayers.size()];
				//Compute all raw values.
				for(int c = 0; c < allLayers.size(); c++) {
					//Current layer handling the computation.
					Layer currentLayer = allLayers.get(c);
					//Current raw output being calculated.
					double[] currentRawOutput = currentLayer.getRawOutput(prev);
					//Save the raw output for the current layer.
					allRawOutputs[c] = currentRawOutput;
					//Activate the raw output for the next iteration.
					prev = currentLayer.applyNonLinearFunction(currentRawOutput);
				}
				//Predicted output is the last computation in the above loop.
				double[] predictedOutput = prev;
				//Create the array to store the derivatives in the last layer. Each output will have a corresponding derivative.
				derivatives[allLayers.size() - 1] = new double[predictedOutput.length];
				//Get the final layer.
				Layer lastLayer = allLayers.get(allLayers.size() - 1);
				//Get the base bias derivative.
				biasAdjustment[allLayers.size() - 1] = 0;
				//Calculate each base case and store it. Operation uses the last layer. dCost/dActivation * dActivation/dRaw is stored.
				for(int c = 0; c < predictedOutput.length; c++) {
					derivatives[allLayers.size() - 1][c] = lastLayer.dCostByDRaw(expectedOutputs[b][c], allRawOutputs[allLayers.size() - 1][c]);
					//Compute the bias derivative.
					biasDerivatives[allLayers.size() - 1] = biasDerivatives[allLayers.size() - 1] + derivatives[allLayers.size() - 1][c];
				}
				//We can also create the derivatives for a Generative Adversarial Network here by changing the expected output value to what was not expected. The value that the generator would want to fool the classifier.
				//For each row in the weight matrix, the column corresponds with the previous activation row, when in vertical vector form.
				//To finish off with dRaw/dWeight, multiply by previous activation.
				//Loop to calculate weight adjustments.
				for(int c = allLayers.size() - 1; c >= 0; c--) {
					//Fetch current layer.
					Layer currentLayer = allLayers.get(c);
					//Fetching current weight matrix.
					double[][] currentWeightMatrix = currentLayer.weightMatrix;
					//Current adjustment matrix.
					double[][] currentAdjustmentMatrix = adjustmentMatrices[c];
					//The array of required derivatives.
					double[] currentDerivatives = derivatives[c];
					//Previous raw values.
					double[] previousRawValues = null;
					//Previous layer to calculate the activation.
					Layer previousLayer = null;
					if(c > 0) {
						previousRawValues = allRawOutputs[c - 1];
						previousLayer = allLayers.get(c - 1);
					}
					else {
						//Special case where if we are at the first layer the previous raw values are the input values.
						previousRawValues = currentInput;
					}
					//Nested for loop to go through all the weights in the layer.
					for(int d = 0; d < currentWeightMatrix.length; d++) {
						//Current row of the matrix.
						double[] currentRow = currentWeightMatrix[d];
						//Loop through current row of matrix.
						for(int e = 0; e < currentRow.length; e++) {
							//Derivative of cost with respect to current weight is multiply the previous tracked derivative by the previous activation.
							if(previousLayer != null) {
								currentAdjustmentMatrix[d][e] = currentAdjustmentMatrix[d][e] + currentDerivatives[d] * previousLayer.applyNonLinearFunction(previousRawValues[e]);
							}
							else {
								//If we are at the first layer.
								currentAdjustmentMatrix[d][e] = currentAdjustmentMatrix[d][e] + currentDerivatives[d] * previousRawValues[e];
							}
							//Bias adjustment.
							biasAdjustment[c] = biasAdjustment[c] + currentDerivatives[d];
						}
					}
					
					if(previousLayer != null) {
						//Construct the derivative bases for the previous layer.
						//The size of the previous layer's derivative vector is the current layers input size.
						derivatives[c - 1] = new double[currentLayer.inputSize];
						//previous derivative vector to be populated.
						double[] previousDerivatives = derivatives[c - 1];
						//The derivative of the bias in the previous layer.
						double previousBiasDerivative = 0;
						//Array of previous raw values.
						previousRawValues = allRawOutputs[c - 1];
						//Loop to populate the derivative vector.
						for(int d = 0; d < previousDerivatives.length; d++) {
							//Derivative of dCost/dPreviousActivation
							double previousDerivative = 0;
							//Summation of current weights multiplied by respective current dCost/dRawValues.
							for(int e = 0; e < currentWeightMatrix.length; e++) {
								//Entries of the derivative vector corresponds to the column value of the weight matrix in terms of the weight that needs to be multiplied.
								//This is also the index of the current corresponding derivative.
								int colVal = d;
								//Summate the products.
								previousDerivative = previousDerivative + currentDerivatives[e] * currentWeightMatrix[e][colVal];
							}
							//Compute the previous bias derivative by adding the regular derivative.
							previousBiasDerivative = previousBiasDerivative + previousDerivative;
							//Perform a final calculation of the derivation by applying the derived function on the raw value and storing it.
							previousDerivatives[d] = previousDerivative * previousLayer.applyDerivedNonLinearFunction(previousRawValues[d]);
							//Update the derivative.
							biasDerivatives[c - 1] = previousBiasDerivative;
						}
					}
				}
			}
			for(int b = 0; b < allLayers.size(); b++) {
				//Fetch current layer.
				Layer currentLayer = allLayers.get(b);
				//Current weight matrix.
				double[][] currentWeightMatrix = currentLayer.weightMatrix;
				//Loop through the weight matrix.
				for(int c = 0; c < currentWeightMatrix.length; c++) {
					//Current matrix row.
					double[] currentWeightRow = currentWeightMatrix[c];
					//Loop through the row.
					for(int d = 0; d < currentWeightRow.length; d++) {
						//Adjust the weight based on the average of all designed nudges multiplied by the learning rate.
						currentWeightRow[d] = currentWeightRow[d] - adjustmentMatrices[b][c][d] * learningRate / inputs.length;
					}
				}
				//Adjust the bias based on the average of all designed nudges multiplied by the learning rate.
				currentLayer.bias = currentLayer.bias - biasAdjustment[b] * learningRate / inputs.length;
			}
		}
	}
}
