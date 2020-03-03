
abstract class SigmoidLayer extends Layer {

	SigmoidLayer() {
		super();
	}

	double[] getOutput(double[] i) {
		double[] input = modifies(i);
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
				toReturn[a] = sigmoid(toReturn[a]);
			}
			return toReturn;
		}
		if(input.length != this.size()) {
			return null;
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
			toReturn[a] = sigmoid(toReturn[a]);
		}
		return toReturn;
	}

	//Make a function that will use this to adjust the weight!
	private double dCostByDWeightSig(int parentNode, int parentNodeEdge, double[] expected, double[] input) {
		double currentWeight = this.parentLayer.nodes.get(parentNode).multipliers[parentNodeEdge];
		double toReturn = 0;
		double outBeforeAct = this.parentLayer.beforeActivator(input, parentNodeEdge);
		for(int a = 0; a < this.nodes.size(); a++) {
			toReturn = toReturn - (NeuralNet.calculateLoss(expected[a], input[a])) * Math.pow(Math.E, outBeforeAct) * currentWeight / Math.pow(Math.pow(Math.E, outBeforeAct) + 1, 2);
		}
		return toReturn;
	}

	public double getNewWeightSig(int parentNode, int parentNodeEdge, double[] expected, double[] input, double rate) {
		double grad = dCostByDWeightSig(parentNode, parentNodeEdge, expected, input);
		double step = stepSize(grad, rate);
		double currentWeight = this.parentLayer.nodes.get(parentNode).multipliers[parentNodeEdge];
		return currentWeight - step;
	}
}
