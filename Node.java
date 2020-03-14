
public class Node {
	double[] edgeWeights;
	
	public Node(int size) {
		this.edgeWeights = new double[size];
		for(int a = 0; a < size; a++) {
			this.edgeWeights[a] = 1.0;
		}
	}
	
	public int size() {
		return this.edgeWeights.length;
	}
	
	public double[] getWeights() {
		return this.edgeWeights.clone();
	}
	
	public void setWeight(int edge, double weight) {
		this.edgeWeights[edge] = weight;
	}
}
