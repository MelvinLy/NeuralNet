import java.util.ArrayList;

abstract class Layer {
	protected ArrayList<Node> nodes;
	protected Layer nextLayer;
	protected int degree;
	
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

	abstract double[] getOutputs();
	
	protected Layer() {
		this.nodes = new ArrayList<Node>();
		this.nextLayer = null;
	}
	
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

	//
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
	
	Node[] getMultipliers() {
		return (Node[]) this.multipliers.toArray();
	}
	
	int getNumOuts() {
		return multipliers.size();
	}

}