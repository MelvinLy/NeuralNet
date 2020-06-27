import java.util.Stack;

public class NeuralNetwork {

	public Layer firstLayer;
	private Layer lastLayer;
	private int size;
	private int inputSize;
	
	//Network can have a single layer as input can be omitted;
	//The output layer is just the output of the last layer;
	public NeuralNetwork(Layer firstLayer) {
		this.firstLayer = firstLayer;
		this.lastLayer = firstLayer;
		this.size = 1;
	}
	
	/*
	public void fit(double[][] inputs, double[][] outputs, int trainAmount, double learningRate) throws UnsupportedMethodException {
		for(int a = 0; a < trainAmount; a++) {
			for(int b = 0; b < inputs.length; b++) {
				fit(inputs[b], outputs[b], 1, learningRate);
			}
		}
	}
	*/
	
	public int getInputSize() {
		return this.inputSize;
	}
	
	public void fit(double[][] inputs, double[][] expected, int trainAmount, double learningRate) throws UnsupportedMethodException {
		this.inputSize = inputs.length;
		for(int a = 0; a < inputs.length; a++) {
			double[][] allOutputs = this.getAllOutputs(inputs[a]);
			
		}
	}
	
	/*
	public void fit(double[] input, double[] output, int trainAmount, double learningRate) throws UnsupportedMethodException {
		if(trainAmount == 0) {
			return;
		}
		double[][] allOutputs = getAllOutputs(input);
		Stack<Layer> s = new Stack<Layer>();
		Layer current = firstLayer;
		while(current!= null) {
			s.push(current);
			current = current.nextLayer;
		}
		while(s.size() > 1) {
			current = s.pop();
			int index = s.size() - 1;
			double[] cInput = allOutputs[index];
			double[] currentExpectedOutput = allOutputs[index];
			if(index == allOutputs.length - 2) {
				currentExpectedOutput = output;
			}
			for(int a = 0; a < trainAmount; a++) {
				double[] rawOut = current.getRawOutput(cInput);
				double[] activatedOut = current.getActivatedOutput(cInput);
				current.trainLayer(activatedOut, currentExpectedOutput, learningRate, rawOut);
				activatedOut = current.getActivatedOutput(cInput);
				allOutputs[index + 1] = activatedOut;
			}
		}
		current = s.pop();
		int index = s.size();
		double[] cInput = input;
		double[] currentExpectedOutput = allOutputs[index];
		for(int a = 0; a < trainAmount; a++) {
			double[] rawOut = current.getRawOutput(cInput);
			double[] activatedOut = current.getActivatedOutput(cInput);
			current.trainLayer(activatedOut, currentExpectedOutput, learningRate, rawOut);
			activatedOut = current.getActivatedOutput(cInput);
			allOutputs[index + 1] = activatedOut;
		}
	}
	*/

	public double[] getOutput(double[] input) {
		double[] out = input;
		Layer currentLayer = firstLayer;
		for(int a = 0; a < size; a++) {
			out = currentLayer.getActivatedOutput(out);
			currentLayer = currentLayer.nextLayer;
		}
		return out;
	}

	//Returns output at each layer
	public double[][] getAllOutputs(double[] input) {
		double[] out = input;
		double[][] toReturn = new double[size][];
		Layer currentLayer = firstLayer;
		for(int a = 0; a < size; a++) {
			out = currentLayer.getActivatedOutput(out);
			currentLayer = currentLayer.nextLayer;
			toReturn[a] = out;
		}
		return toReturn;
	}

	public void addLayer(Layer layer) throws LayerSizeMismatchException {
		if(layer.inputSize != lastLayer.outputSize) {
			throw new LayerSizeMismatchException("The current last layer's output size is not equal to the given layer's input size.");
		}
		this.lastLayer.nextLayer = layer;
		this.lastLayer = lastLayer.nextLayer;
		this.size = this.size + 1;
	}
}
