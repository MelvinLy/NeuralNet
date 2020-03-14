public class Node {
	
	private double[] edgeWeights;
	
	//Size correlates to the amount of edges coming out of the node.
	public Node(int size) {
		this.edgeWeights = new double[size];
		//Sets default weight to 1.
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
	
	//Set the weight of each edge.
	public void setWeight(int edge, double weight) {
		this.edgeWeights[edge] = weight;
	}
}
