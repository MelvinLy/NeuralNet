abstract class ReLULayer extends Layer {

	ReLULayer() {
		super();
	}

	double[] getOutput(double[] i) throws LayerMismatchException {
		double[] input = modifies(i);
		if(!this.bias) {
			if(input.length != this.size()) {
				throw new LayerMismatchException("Input does not match layer size.");
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
				toReturn[a] = reLU(toReturn[a]);
			}
			return toReturn;
		}
		if(input.length != this.size()) {
			throw new LayerMismatchException("Input does not match layer size.");
		}
		double[] biases = this.getBiasMult();
		Node[] nodes = this.getNodes();
		double[] toReturn = new double[nodes[0].getNumOuts()];
		for(int a = 0; a < nodes.length; a++) {
			Node currentNode = nodes[a];
			for(int b = 0; b < currentNode.getNumOuts(); b++) {
				toReturn[b] = toReturn[b] + input[a] * currentNode.getMultipliers()[b];
			}
			toReturn[a] = toReturn[a] + biases[a];
		}
		for(int a = 0; a < toReturn.length; a++) {
			toReturn[a] = reLU(toReturn[a]);
		}
		return toReturn;
	}

	//Max depth is size of net - 1;
	public double dCostByDWeight(NeuralNet net, int depth, int parentNode, int parentNodeEdge, double[] expected, double[] inputs) throws LayerMismatchException {
		double toReturn = 0;
		double[] output = net.getOutput(inputs, depth);
		double[] temp = net.getOutput(inputs, depth - 1);
		double outBeforeAct = this.parentLayer.beforeActivator(temp, parentNodeEdge);
		if(this.parentLayer instanceof SigmoidLayer) {
			for(int a = 0; a < this.nodes.size(); a++) {		
				toReturn = toReturn - ((expected[a] - output[a]) * Math.pow(Math.E, outBeforeAct) * inputs[parentNode] / Math.pow(Math.pow(Math.E, outBeforeAct), 2));
			}
		}
		else {
			for(int a = 0; a < this.nodes.size(); a++) {
				if(output[a] != 0) {
					toReturn = toReturn - ((expected[a] - output[a]) * inputs[parentNode]);
				}
			}
		}
		return toReturn;
	}

	public double getNewWeight(NeuralNet net, int depth, int parentNode, int parentNodeEdge, double[] expected, double[] inputs, double rate) throws LayerMismatchException {
		double grad = this.dCostByDWeight(net, depth, parentNode, parentNodeEdge, expected, inputs);
		double step = this.stepSize(grad, rate);
		double currentWeight = this.parentLayer.nodes.get(parentNode).multipliers[parentNodeEdge];
		return currentWeight - step * rate;
	}
}
