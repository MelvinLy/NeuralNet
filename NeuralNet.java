import java.util.*;

public class NeuralNet {


}

class Node {	
	int outs;
	ArrayList<Double> multipliers;
	
	Node() {
		this.multipliers = new ArrayList<Double>();
		this.outs = 0;
	}
	
	Node[] getMultipliers() {
		return (Node[]) this.multipliers.toArray();
	}
	
	int getOuts() {
		return outs;
	}
	
	boolean addMultiplier(double k) {
		if(k > 1 || k < -1) {
			return false;
		}
		this.multipliers.add(k);
		this.outs++;
		return true;
	}
}

abstract class Layer {
	ArrayList<Node> nodes;
	Layer nextLayer;
	
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
}

class ReLULayer extends Layer {


	double[] getOutputs() {
		
		return null;
	}
	
}
