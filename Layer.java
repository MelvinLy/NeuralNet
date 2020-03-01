import java.security.InvalidParameterException;
import java.util.*;

abstract class Layer {
	private ArrayList<Node> nodes;
	private Layer nextLayer;
	private double[] biasMult;
	protected boolean bias;
	
	static double sigmoid(double x) {
		
		double nom = Math.pow(Math.E, x);
		double den = Math.pow(Math.E, x) + 1;
		return nom / den;
		/*
		double nom = Math.pow(Math.E, k * x) - 1;
		double den = Math.pow(Math.E, k * x) + 1;
		return nom / den;
		*/
	}
	
	static double sigmoidPrime(double x) {
		double nom = Math.pow(Math.E, x);
		double den = Math.pow(Math.pow(Math.E, x) + 1, 2);
		return nom / den;
	}
	
	static double reLU(double x) {
		if(x >= 0) {
			return x;
		}
		return 0;
	}

	protected Layer() {
		this.nodes = new ArrayList<Node>();
		this.nextLayer = null;
		this.bias = false;
	}
	
	abstract double[] getOutput(double[] input);
	
	void enableBias() {
		this.bias = true;
	}
	
	void disableBias() {
		this.bias = false;
	}
	
	public static Node createNode(int outs) {
		return new Node(outs);
	}
	
	//The amount of nodes in the neural network.
	protected int size() {
		return this.nodes.size();
	}
	
	//Nodes must have same outs to be added.
	protected void addNode(Node node) throws NodeSizeMismatchException {
		if(this.nodes.size() == 0) {
			this.nodes.add(node);
			return;
		}
		if(this.nodes.get(0).getNumOuts() == node.getNumOuts()) {
			this.nodes.add(node);
			return;
		}
		throw new NodeSizeMismatchException("The node being added does not match the size of the read of the nodes in the layer.");
	}
	
	protected void setMult(int i, double k) throws NoEdgeException {
		if(this.biasMult != null) {
			this.biasMult[i] = k;
			return;
		}
		throw new NoEdgeException("The corresponding edge of i does not exist.");
	}
	
	protected double[] getMult() {
		return biasMult.clone();
	}
	
	public Node[] getNodes() {
		Node[] toReturn = new Node[this.nodes.size()];
		for(int a = 0; a < toReturn.length; a++) {
			toReturn[a] = this.nodes.get(a);
		}
		return toReturn;
	}

	//The next layer size must have the same outs as the nodes in this layer.
	protected void addNextLayer(Layer nextLayer) throws LayerMismatchException, EmptyLayerException {
		if(this.size() == 0) {
			throw new EmptyLayerException("The current layer is empty and cannot add a new layer.");
		}
		if(nextLayer.size() != this.nodes.get(0).getNumOuts()) {
			throw new LayerMismatchException("Size of output for this layer does not match input of layer to be added.");
		}
		this.nextLayer = nextLayer;
		this.biasMult = new double[this.nextLayer.size()];
	}
	
	protected Layer getNextLayer() {
		return this.nextLayer;
	}
	
	protected boolean hasBias() {
		return bias;
	}
}

class Node {	
	double[] multipliers;
	
	Node(int size) {
		this.multipliers = new double[size];
	}
	
	double[] getMultipliers() {
		return this.multipliers.clone();
	}
	
	int getNumOuts() {
		return multipliers.length;
	}

	void setMultiplier(int i, double m) {
		if(m > 1 || m < -1) {
			throw new InvalidParameterException("Value must be between 1 and -1, inclusive.");
		}
		this.multipliers[i] = m;
	}
	
	public String toString() {
		return Arrays.toString(this.multipliers);
	}
}