import java.util.ArrayList;

abstract class Layer {
	private ArrayList<Node> nodes;
	private Layer nextLayer;
	private double[] biasMult;
	protected double bias;
	
	static double sigmoid(double x, double k) {
		double nom = Math.pow(Math.E, k * x) - 1;
		double den = Math.pow(Math.E, k * x) + 1;
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
		this.bias = 0;
	}
	
	abstract double[] getOutput(double[] input);
	
	void enableBias() {
		this.bias = 1;
	}
	
	void disableBias() {
		this.bias = 0;
	}
	
	public static Node createNode(int outs) {
		return new Node(outs);
	}
	
	//The amount of nodes in the neural network.
	protected int size() {
		return this.nodes.size();
	}
	
	//Nodes must have same outs to be added.
	protected boolean addNode(Node node) {
		if(this.nodes.size() == 0) {
			this.nodes.add(node);
			return true;
		}
		if(this.nodes.get(0).getNumOuts() == node.getNumOuts()) {
			this.addNode(node);
			return true;
		}
		return false;
	}
	
	protected boolean setMult(int i, double k) {
		if(this.biasMult != null) {
			this.biasMult[i] = k;
			return true;
		}
		return false;
	}
	
	protected double[] getMult() {
		return biasMult.clone();
	}
	
	protected Node[] getNodes() {
		return (Node[]) this.nodes.toArray();
	}

	//The next layer size must have the same outs as the nodes in this layer.
	protected boolean addNextLayer(Layer nextLayer) {
		if(this.size() == 0) {
			return false;
		}
		if(this.nextLayer.size() != this.nodes.get(0).getNumOuts()) {
			return false;
		}
		this.nextLayer = nextLayer;
		this.biasMult = new double[this.nextLayer.size()];
		return true;
	}
	
	protected Layer getNextLayer() {
		return this.nextLayer;
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

	boolean setMultiplier(int i, double m) {
		if(m > 1 || m < -1) {
			return false;
		}
		this.multipliers[i] = m;
		return true;
	}
}