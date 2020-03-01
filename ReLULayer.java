
class ReLULayer extends Layer {

	ReLULayer() {
		super();
	}
	
	double[] getOutput(double[] input) {
		if(input.length != this.size()) {
			return null;
		}
		Node[] nodes = this.getNodes();
		double[] toReturn = new double[nodes.length];
		for(int a = 0; a < nodes.length; a++) {
			Node currentNode = nodes[a];
			for(int b = 0; b < currentNode.getNumOuts(); b++) {
				toReturn[a] = toReturn[a] + input[a] * currentNode.getMultipliers()[b];
			}
		}
		for(int a = 0; a < toReturn.length; a++) {
			toReturn[a] = reLU(toReturn[a]);
		}
		return toReturn;
	}
}
