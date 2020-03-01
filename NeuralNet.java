import java.util.ArrayList;

public class NeuralNet {
	
	public static double sigmoid(float x, float k) {
		double nom = Math.pow(Math.E, k * x) - 1;
		double den = Math.pow(Math.E, k * x) + 1;
		return nom / den;
	}
	
}

class Node {
	private ArrayList<Node> children;
	private ArrayList<Double> multiplier;
	private double k;
	
	static double sigmoid(double x, double k) {
		double nom = Math.pow(Math.E, k * x) - 1;
		double den = Math.pow(Math.E, k * x) + 1;
		return nom / den;
	}
	
	
	Node(double k) {
		this.children = new ArrayList<Node>();
		this.multiplier = new ArrayList<Double>();
		this.k = k;
	}
	
	ArrayList<Node> getChild()  {
		return (ArrayList<Node>) this.children.clone();
	}
	
	ArrayList<Double> getMultiplier() {
		return (ArrayList<Double>) this.multiplier.clone();
	}
	
	double getK() {
		return this.k;
	}
	
}
