import java.util.ArrayList;
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
		//Add the layer to the List of all ordered layers. This uses only a clone of the given layer.
		allLayers.add(layer.clone());
	}

	//Replace and existing layer.
	public Layer replaceLayer(int layerNumber, Layer layer) throws LayerDoesNotExistException, LayerSizeMismatchException {
		//Throw an error if given layer is out of range.
		if(layerNumber < 0 || layerNumber >= allLayers.size()) {
			throw new LayerDoesNotExistException("The given layer does not exist.");
		}
		//Throw an error if the input and output size does not match the existing one.
		if(this.allLayers.get(layerNumber).getInputSize() != layer.getInputSize() || this.allLayers.get(layerNumber).getOutputSize() != layer.getOutputSize()) {
			throw new LayerSizeMismatchException("The give layer input and output size does match the existing one.");
		}
		//Returns the previous layer that was their.
		return this.allLayers.set(layerNumber, layer);
	}

	//Creates an empty network.
	public NeuralNetwork() {
		this.allLayers = new ArrayList<Layer>();
	}

	//Returns a clone of the desire layer.
	public Layer getLayer(int layerNumber) throws LayerDoesNotExistException {
		//Throw an error if given layer is out of range.
		if(layerNumber < 0 || layerNumber >= allLayers.size()) {
			throw new LayerDoesNotExistException("The given layer does not exist.");
		}
		//Return the copy of the layer.
		return allLayers.get(layerNumber).clone();
	}

	//Method that clones from a desired inclusive starting layer to a desired exclusive final layer.
	public NeuralNetwork clone(int start, int end) throws LayerSizeMismatchException {
		//Create the new network.
		NeuralNetwork out = new NeuralNetwork();
		//Return empty network if there are invalid inputs.
		if(start < 0) {
			return out;
		}
		else if(end > this.allLayers.size()) {
			return out;
		}
		//Loop through layers.
		for(int a = start; a < end; a++) {
			//Current layer to be cloned.
			Layer currentLayer = this.allLayers.get(a);
			//Cloning the layers.
			out.addLayer(currentLayer.clone());
		}
		//Return the new network.
		return out;
	}

	@SuppressWarnings("unchecked")
	public NeuralNetwork clone() {
		//Create a new network.
		NeuralNetwork out = new NeuralNetwork();
		out.allLayers = (ArrayList<Layer>) this.allLayers.clone();
		return out;
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
		//Check if the network has any layers.
		if(allLayers.size() == 0) {
			allLayers.add(layer);
			return;
		}
		//Check if the last layers output is equal to the input size of the given layer.
		if(allLayers.get(allLayers.size() - 1).getOutputSize() != layer.getInputSize()) {
			throw new LayerSizeMismatchException("The output size of the last layer is not equal to the input size of the layer given.");
		}
		allLayers.add(layer);
	}

	public double[] getOutputVector(double[] input) throws InputSizeMismatchException, NoLayersException {
		//Check that there are layers in the network.
		if(this.allLayers.size() == 0) {
			throw new NoLayersException("There are no layers in the network.");
		}
		if(input.length != allLayers.get(0).getInputSize()) {
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

	//Get the total amount of layers.
	public int getTotalLayers() {
		return this.allLayers.size();
	}

	//Manually fit the network with a given gradient set.
	//Applies the changes from the starting layer inclusive and stops at ending layer exclusive.
	public void manualFit(double[][][] weightGradients, double[][] biasGradients, double learningRate, int startingLayer, int endingLayer) throws LayerSizeMismatchException {
		//Index of weightGradients and biasGradients.
		int i = 0;
		if(endingLayer - startingLayer > weightGradients.length || endingLayer - startingLayer > biasGradients.length) {
			throw new LayerSizeMismatchException("There are not enough gradients to match the amount of layer.");
		}
		//Loop to apply the needed adjustments to the weight values.
		for(int a = startingLayer; a < endingLayer; a++, i++) {
			//Fetch current layer.
			Layer currentLayer = allLayers.get(a);
			//Current weight matrix.
			double[][] currentWeightMatrix = currentLayer.weightMatrix;
			//Current biases for the layer.
			double[] currentBiases = currentLayer.biases;		
			//Check if the current layer and weight matrix match in size.
			if(currentLayer.getInputSize() != weightGradients[i][0].length || currentLayer.getOutputSize() != weightGradients[i].length) {
				throw new LayerSizeMismatchException("Layer number "  + a + "'s size does not match with the current gradient matrix size.");
			}
			else if(biasGradients[i].length != currentLayer.biases.length) {
				throw new LayerSizeMismatchException("Layer number " + a + "'s size does not match with the current bias vectpr size.");
			}
			//Loop through the weight matrix. The amount of rows is equal to the output size and the amount of biases there are in the layer.
			for(int b = 0; b < currentWeightMatrix.length; b++) {
				//Current matrix row.
				double[] currentWeightRow = currentWeightMatrix[b];
				//Loop through the row.
				for(int c = 0; c < currentWeightRow.length; c++) {
					//Adjust the weight based on the average of all designed nudges multiplied by the learning rate.
					currentWeightRow[c] = currentWeightRow[c] - weightGradients[i][b][c] * learningRate;
				}
				//Update the currentBiases.
				currentBiases[b] = currentBiases[b] - biasGradients[i][b] * learningRate;
			}
		}
	}
	
	//Similar to fit, but does not apply the adjustments but returns the positive gradient vector.
	//The end index is inclusive right to left.
	public Object[] getGradients(double[][] inputs, double[][] expectedOutputs, int end) throws NoLayersException, OutputSizeMismatchException, InputSizeMismatchException, LayerDoesNotExistException {
		//Check that there are layers in the network.
		if(this.allLayers.size() == 0) {
			throw new NoLayersException("There are no layers in the network.");
		}
		//Check if the ending layer is out of bounds.
		if(end < 0 || end >= this.getTotalLayers()) {
			throw new LayerDoesNotExistException("The ending layer does not exist.");
		}
		//Collection of all weight adjustments averaged. Positive gradient at the moment.
		double[][][] adjustmentMatrices = new double[this.getTotalLayers() - end][][];
		//Create the bias adjustment matrix.
		double[][] biasAdjustmentMatrix =  new double[this.getTotalLayers() - end][];
		//The index for adjustmentMatrices and biasAdjustmustmentMatrix.
		int i = adjustmentMatrices.length - 1;
		//Loop through to create appropriate size of the weight adjustment and bias adjustment for each layer.
		//Create the 3D adjustment matrix.
		for(int c = allLayers.size() - 1; c >= end; c--) {
			//Fetch the current layer.
			Layer currentLayer = allLayers.get(c);
			//Create the gradient adjustment matrix at the current layer.
			adjustmentMatrices[i] = new double[currentLayer.weightMatrix.length][currentLayer.weightMatrix[0].length];
			//The bias adjustment array in respect to the current layer is the output size.
			biasAdjustmentMatrix[i--] = new double[currentLayer.getOutputSize()];
		}
		//Looping through all given inputs as training data.
		for(int b = 0; b < inputs.length; b++) {
			//Check that the corresponding output has the expected network output size.
			if(expectedOutputs[b].length != this.allLayers.get(allLayers.size() - 1).getOutputSize()) {
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
			//Calculate each base case and store it. Operation uses the last layer. dCost/dActivation * dActivation/dRaw is stored.
			for(int c = 0; c < predictedOutput.length; c++) {
				derivatives[allLayers.size() - 1][c] = lastLayer.dCostByDRaw(expectedOutputs[b][c], allRawOutputs[allLayers.size() - 1][c]);
			}
			//The index for adjustmentMatrices and biasAdjustmustmentMatrix.
			i = adjustmentMatrices.length - 1;
			//We can also create the derivatives for a Generative Adversarial Network here by changing the expected output value to what was not expected. The value that the generator would want to fool the classifier.
			//For each row in the weight matrix, the column corresponds with the previous activation row, when in vertical vector form.
			//To finish off with dRaw/dWeight, multiply by previous activation.
			//Loop to calculate weight adjustments.
			for(int c = allLayers.size() - 1; c >= end; c--) {
				//Fetch current layer.
				Layer currentLayer = allLayers.get(c);
				//Fetching current weight matrix.
				double[][] currentWeightMatrix = currentLayer.weightMatrix;
				//Current adjustment matrix.
				double[][] currentAdjustmentMatrix = adjustmentMatrices[i];
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
				//Current layer's bias adjustments.
				double[] currentBiasAdjustments = biasAdjustmentMatrix[i--];
				//Nested for loop to go through all the weights in the layer. The amount of rows is equal to the output size and the amount of biases there are in the layer.
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
					}
					currentBiasAdjustments[d] = currentBiasAdjustments[d] + currentDerivatives[d];
				}
				if(previousLayer != null) {
					//Construct the derivative bases for the previous layer.
					//The size of the previous layer's derivative vector is the current layers input size.
					derivatives[c - 1] = new double[currentLayer.getInputSize()];
					//previous derivative vector to be populated.
					double[] previousDerivatives = derivatives[c - 1];
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
						//Perform a final calculation of the derivation by applying the derived function on the raw value and storing it.
						previousDerivatives[d] = previousDerivative * previousLayer.applyDerivedNonLinearFunction(previousRawValues[d]);
					}
				}
			}
		}
		//Index for adjustmentMatrices and biasAdjustmentMatrix.
		i = 0;
		//Loop to apply the average.
		for(int b = end; b < allLayers.size(); b++, i++) {
			//Fetch current layer.
			Layer currentLayer = allLayers.get(b);
			//Current weight matrix.
			double[][] currentWeightMatrix = currentLayer.weightMatrix;
			//Loop through the weight matrix. The amount of rows is equal to the output size and the amount of biases there are in the layer.
			for(int c = 0; c < currentWeightMatrix.length; c++) {
				//Current matrix row.
				double[] currentWeightRow = currentWeightMatrix[c];
				//Loop through the row.
				for(int d = 0; d < currentWeightRow.length; d++) {
					//Apply the average.
					adjustmentMatrices[i][c][d] = adjustmentMatrices[i][c][d] / inputs.length;
				}
				//Apply the average.
				biasAdjustmentMatrix[i][c] = biasAdjustmentMatrix[i][c] / inputs.length;
			}
		}
		//First index contains the adjustment matrix for the weights. Second index contains the adjustment matrix for the biases.
		Object[] out = new Object[2];
		out[0] = adjustmentMatrices;
		out[1] = biasAdjustmentMatrix;
		return out;
	}

	//Create a neural network model.
	public void fit(double[][] inputs, double[][] expectedOutputs, int trainCycles, double learningRate) throws InputSizeMismatchException, OutputSizeMismatchException, NoLayersException {
		//Check that there are layers in the network.
		if(this.allLayers.size() == 0) {
			throw new NoLayersException("There are no layers in the network.");
		}
		for(int a = 0; a < trainCycles; a++) {
			//Collection of all weight adjustments averaged. Positive gradient at the moment.
			double[][][] adjustmentMatrices = new double[allLayers.size()][][];
			//Create the bias adjustment matrix.
			double[][] biasAdjustmentMatrix =  new double[allLayers.size()][];
			//Loop through to create appropriate size of the weight adjustment and bias adjustment for each layer.
			//Create the 3D adjustment matrix.
			for(int c = allLayers.size() - 1; c >= 0; c--) {
				//Fetch the current layer.
				Layer currentLayer = allLayers.get(c);
				//Create the gradient adjustment matrix at the current layer.
				adjustmentMatrices[c] = new double[currentLayer.weightMatrix.length][currentLayer.weightMatrix[0].length];
				//The bias adjustment array in respect to the current layer is the output size.
				biasAdjustmentMatrix[c] = new double[currentLayer.getOutputSize()];
			}
			//Looping through all given inputs as training data.
			for(int b = 0; b < inputs.length; b++) {
				//Check that the corresponding output has the expected network output size.
				if(expectedOutputs[b].length != this.allLayers.get(allLayers.size() - 1).getOutputSize()) {
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
				//Calculate each base case and store it. Operation uses the last layer. dCost/dActivation * dActivation/dRaw is stored.
				for(int c = 0; c < predictedOutput.length; c++) {
					derivatives[allLayers.size() - 1][c] = lastLayer.dCostByDRaw(expectedOutputs[b][c], allRawOutputs[allLayers.size() - 1][c]);
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
					//Current layer's bias adjustments.
					double[] currentBiasAdjustments = biasAdjustmentMatrix[c];
					//Nested for loop to go through all the weights in the layer. The amount of rows is equal to the output size and the amount of biases there are in the layer.
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
						}
						currentBiasAdjustments[d] = currentBiasAdjustments[d] + currentDerivatives[d];
					}
					if(previousLayer != null) {
						//Construct the derivative bases for the previous layer.
						//The size of the previous layer's derivative vector is the current layers input size.
						derivatives[c - 1] = new double[currentLayer.getInputSize()];
						//previous derivative vector to be populated.
						double[] previousDerivatives = derivatives[c - 1];
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
							//Perform a final calculation of the derivation by applying the derived function on the raw value and storing it.
							previousDerivatives[d] = previousDerivative * previousLayer.applyDerivedNonLinearFunction(previousRawValues[d]);
						}
					}
				}
			}
			//Loop to apply the needed adjustments to the weight values.
			for(int b = 0; b < allLayers.size(); b++) {
				//Fetch current layer.
				Layer currentLayer = allLayers.get(b);
				//Current weight matrix.
				double[][] currentWeightMatrix = currentLayer.weightMatrix;
				//Current biases for the layer.
				double[] currentBiases = currentLayer.biases;
				//Loop through the weight matrix. The amount of rows is equal to the output size and the amount of biases there are in the layer.
				for(int c = 0; c < currentWeightMatrix.length; c++) {
					//Current matrix row.
					double[] currentWeightRow = currentWeightMatrix[c];
					//Loop through the row.
					for(int d = 0; d < currentWeightRow.length; d++) {
						//Adjust the weight based on the average of all designed nudges multiplied by the learning rate.
						currentWeightRow[d] = currentWeightRow[d] - adjustmentMatrices[b][c][d] * learningRate / inputs.length;
					}
					//Update the currentBiases.
					currentBiases[c] = currentBiases[c] - biasAdjustmentMatrix[b][c] * learningRate / inputs.length;
				}
			}
		}
	}
}

