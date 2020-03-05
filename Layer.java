import java.util.*;

public abstract class Layer {
	protected Node[] nodes;
	protected Layer parentLayer;
	protected Layer nextLayer;
	protected NeuralNetwork parentNetwork;
	protected int outputSize;
	protected int layerNumber;
	private double bias;
	
	public Layer(int size, int outputSize) {
		this.nodes = new Node[size];
		this.nextLayer = null;
		this.bias = 0;
		this.outputSize = outputSize;
	}
	
	//Take input before activation.
	//Edge defines the output node also.
	public abstract double dCostbyDWeight(int node, int edge, double[] input, double[] expected) throws NullNodeException;
	
	//Also known as get output.
	public abstract double[] activate(double[] input);
	
	public double[] getOutputBeforeAct(double[] input) throws NullNodeException {
		double[] toReturn = new double[this.getOutputSize()];
		//Nodes
		for(int a = 0; a < this.size(); a++) {
			Node currentNode = this.getNodes()[a];
			//Edge
			for(int b = 0; b < this.getOutputSize(); b++) {
				toReturn[b] = toReturn[b] + input[a] * currentNode.getWeights()[b];
			}
		}
		return toReturn;
	}
	
	public double getNewWeight(int node, int edge, double[] input, double[] expected, double rate) throws NullNodeException {
		double[][] allOutput = this.parentNetwork.getAllOutputs(input);
		double[] output = allOutput[this.layerNumber - 1];
		double grad = dCostbyDWeight(node, edge, output, expected);
		double step = grad * rate;
		double previousWeight = this.getNodes()[node].getWeights()[edge];
		return previousWeight - step;
	}
	
	public int size() {
		return this.nodes.length;
	}
	
	public int getOutputSize() {
		return this.outputSize;
	}
	
	public Node[] getNodes() throws NullNodeException {
		for(int a = 0; a < this.nodes.length; a++) {
			if(this.nodes[a] == null) {
				throw new NullNodeException("Not all node(s) in the layer are initialized.");
			}
		}
		return this.nodes.clone();
	}
	
	public Layer getParentLayer() {
		return this.parentLayer;
	}
	
	public Layer getNextLayer() {
		return this.nextLayer;
	}
	
	public double getBias() {
		return this.bias;
	}
	
	public void setNode(int a, Node node) throws NodeSizeMismatchException {
		if(node.size() != this.getOutputSize()) {
			throw new NodeSizeMismatchException("The node being set does not match the output size of the layer.");
		}
		this.nodes[a] = node;
	}
	
	public void setWeight(int node, int edge, double weight) {
		Node temp = this.nodes[node];
		temp.setWeight(edge, weight);
		this.nodes[node] = temp;
	}
	
	public void addNextLayer(Layer layer) throws LayerSizeMismatchException {
		if(layer.size() != this.getOutputSize()) {
			throw new LayerSizeMismatchException("The output size of this current layer is not the same as the next layer.");
		}
		layer.parentLayer = this;
		layer.parentNetwork = this.parentNetwork;
		this.nextLayer = layer;
	}
	
}
