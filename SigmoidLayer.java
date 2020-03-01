
class SigmoidLayer extends Layer {

	private double k;
	
	SigmoidLayer(double k) {
		super();
		this.k = k;
	}
	
	double[] getOutput(double[] input) {
		if(!this.bias) {
			if(input.length != this.size()) {
				return null;
			}
			Node[] nodes = this.getNodes();
			double[] toReturn = new double[nodes[0].getNumOuts()];
			for(int a = 0; a < nodes.length; a++) {
				Node currentNode = nodes[a];
				for(int b = 0; b < currentNode.getNumOuts(); b++) {
					toReturn[b] = toReturn[b] + input[a] * currentNode.getMultipliers()[b];
				}
			}
			for(int a = 0; a < toReturn.length; a++) {
				toReturn[a] = sigmoid(toReturn[a], k);
			}
			return toReturn;
		}
		if(input.length != this.size()) {
			return null;
		}
		Node[] nodes = this.getNodes();
		double[] toReturn = new double[nodes[0].getNumOuts()];
		for(int a = 0; a < nodes.length; a++) {
			Node currentNode = nodes[a];
			for(int b = 0; b < currentNode.getNumOuts(); b++) {
				toReturn[b] = toReturn[b] + input[a] * currentNode.getMultipliers()[b];
			}
			toReturn[a] = toReturn[a] + this.getMult()[a];
		}
		for(int a = 0; a < toReturn.length; a++) {
			toReturn[a] = sigmoid(toReturn[a], k);
		}
		return toReturn;
	}
}
