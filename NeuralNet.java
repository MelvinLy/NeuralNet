import java.util.*;

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
	
	static double reLU(double x) {
		if(x >= 0) {
			return x;
		}
		return 0;
	}
	
	//sum then squash
	
	Node(double k) {
		this.children = new ArrayList<Node>();
		this.multiplier = new ArrayList<Double>();
		this.k = k;
	}
	
	Node[] getChild()  {
		return (Node[]) this.children.toArray();
	}
	
	Node[] getMultiplier() {
		return (Node[]) this.multiplier.toArray();
	}
	
	double getK() {
		return this.k;
	}
	
	
}
