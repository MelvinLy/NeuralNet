import java.util.ArrayList;

abstract class Layer {
	protected ArrayList<Node> nodes;
	protected Layer nextLayer;
	
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
	}
	
	abstract double[] getOutputs(double[] input);
	
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
		return true;
	}
	
}

class Node {	
	ArrayList<Double> multipliers;
	
	Node() {
		this.multipliers = new ArrayList<Double>();

	}
	
	Double[] getMultipliers() {
		return (Double[]) this.multipliers.toArray();
	}
	
	int getNumOuts() {
		return multipliers.size();
	}

}