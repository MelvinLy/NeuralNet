

public class NeuralNet {
	
	public double sigmoid(float x, float k) {
		double nom = Math.pow(Math.E, k * x) - 1;
		double den = Math.pow(Math.E, K * x) + 1;
		return nom / den;
	}
	
}

class Node {
	private Node parent;
	private Node child;
	private double multiplier;
	
	Node(Node parent, Node child, double multiplier) {
		this.parent = parent;
		this.child = child;
		this.multiplier = multiplier;
	}
	
	Node getParent() {
		return this.parent;
	}
	
	Node getChild()  {
		return this.child;
	}
	
	double getMultiplier() {
		return this.multiplier;
	}
	
	void setParent(Node parent) {
		this.parent = parent;
	}
	
	void setChild(Node child) {
		this.child = child;
	}
	
	void setMultiplier(double multiplier) {
		this.multiplier = multiplier;
	}
}
