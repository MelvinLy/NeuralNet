

public class NeuralNet {
	
	public static double sigmoid(float x, float k) {
		double nom = Math.pow(Math.E, k * x) - 1;
		double den = Math.pow(Math.E, k * x) + 1;
		return nom / den;
	}
	
}

class Node {
	private Node parent;
	private Node child;
	private double multiplier;
	private double k;
	
	static double sigmoid(double x, double k) {
		double nom = Math.pow(Math.E, k * x) - 1;
		double den = Math.pow(Math.E, k * x) + 1;
		return nom / den;
	}
	
	
	/**
	 * 
	 * @param parent Parent node.
	 * @param child Child node.
	 * @param multiplier A multiplier.
	 * @param k K used in sigmoid.
	 */
	Node(Node parent, Node child, double multiplier, double k) {
		this.parent = parent;
		this.child = child;
		this.multiplier = multiplier;
		this.k = k;
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
	
	double getK() {
		return this.k;
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
	
	void setK(double k) {
		this.k = k;
	}
}
