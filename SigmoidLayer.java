
class SigmoidLayer extends Layer {

	private double k;
	
	SigmoidLayer(double k) {
		super();
		this.k = k;
	}
	
	double[] getOutputs(double[] input) {
		if(input.length != this.size()) {
			return null;
		}
		Node[] nodes = super.getNodes();
		double[] toReturn = new double[nodes.length];
		for(int a = 0; a < nodes.length; a++) {
			Node currentNode = nodes[a];
			for(int b = 0; b < currentNode.getNumOuts(); b++) {
				toReturn[a] = toReturn[a];
			}
		}
		return null;
	}

}
