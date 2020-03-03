import java.util.Arrays;

abstract class SigmoidLayer extends Layer {

	SigmoidLayer() {
		super();
	}

	double[] getOutput(double[] i) throws LayerMismatchException {
		double[] input = modifies(i.clone());
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
				toReturn[a] = sigmoid(toReturn[a]);
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
			toReturn[a] = sigmoid(toReturn[a]);
		}
		return toReturn;
	}

	//Make a function that will use this to adjust the weight!
	public double dCostByDWeightSig(NeuralNet net, int depth, int parentNode, int parentNodeEdge, double[] expected, double[] inputs) throws LayerMismatchException {
		double toReturn = 0;
		double outBeforeAct = this.parentLayer.beforeActivator(inputs, parentNodeEdge);
		double[] output = net.getOutput(inputs, depth);
		System.out.println("lol " + Arrays.toString(output));
		System.out.println("lol " + Arrays.toString(inputs));
		for(int a = 0; a < this.nodes.size(); a++) {		
			toReturn = toReturn - (NeuralNet.calculateLoss(expected[a], output[a])) * Math.pow(Math.E, outBeforeAct) * inputs[parentNode] / Math.pow(Math.pow(Math.E, outBeforeAct), 2);
		}
		return toReturn;
	}

	public double getNewWeightSig(NeuralNet net, int depth, int parentNode, int parentNodeEdge, double[] expected, double[] inputs, double rate) throws LayerMismatchException {
		double grad = dCostByDWeightSig(net, depth, parentNode, parentNodeEdge, expected, inputs);
		double step = stepSize(grad, rate);
		double currentWeight = this.parentLayer.nodes.get(parentNode).multipliers[parentNodeEdge];
		return currentWeight - step;
	}
}
