import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;

public class Main {
	
	public static void print(Object a) {
		System.out.print(a);
	}
	
	public static int getMax(int[] arr) {
		int max = arr[0];
		for(int a = 1; a < arr.length; a++) {
			if(arr[a] > max) {
				max = arr[a];
			}
		}
		return max;
	}
	
	public static void invert(int[] arr) {
		for(int a = 0; a < arr.length; a++) {
			arr[a] = -arr[a];
		}
	}
	
	public static double[] normalize(int[] arr) {
		double[] out = new double[arr.length];
		int max = getMax(arr);
		for(int a = 0; a < out.length; a++) {
			out[a] = (double) arr[a] / max;
		}
		return out;
	}
	
	public static double round(double val, int places) {
		long mult = (long) Math.pow(10, places);
		return (double) Math.round(val * mult) / mult;
	}
	
	public static void main(String args[]) throws LayerSizeMismatchException, NullNodeException, NodeSizeMismatchException {
		
		double[][] imgArr = new double[24][1024];
		int imgP = 0;
		for(int c = 1; c <= 12; c++) {
			BufferedImage img = null;
			try {
				img = ImageIO.read(new File("one" + c + ".jpg"));
			}
			catch(IOException e) {
				
			}
			int[] pixels = new int[32 * 32];
		
		//Extract pixels
			int pPointer = 0;
			for(int a = 0; a < 32; a++) {
				for(int b = 0; b < 32; b++) {
					pixels[pPointer++] = img.getRGB(b, a);
				}
			}
			invert(pixels);
			double[] normalizedPixels = normalize(pixels);
			for(int a = 0; a < normalizedPixels.length; a++) {
				normalizedPixels[a] = round(normalizedPixels[a], 2);
			}
			/*
			for(int a = 0; a < normalizedPixels.length; a++) {
				if(a % 32 == 0) {
					System.out.println();
				}
				System.out.printf("%.1f ", normalizedPixels[a]);
			}
			*/
			imgArr[imgP++] = normalizedPixels;
		}
		for(int c = 1; c <= 12; c++) {
			BufferedImage img = null;
			try {
				img = ImageIO.read(new File("zero" + c + ".jpg"));
			}
			catch(IOException e) {
				
			}
			int[] pixels = new int[32 * 32];
		
		//Extract pixels
			int pPointer = 0;
			for(int a = 0; a < 32; a++) {
				for(int b = 0; b < 32; b++) {
					pixels[pPointer++] = img.getRGB(b, a);
				}
			}
			invert(pixels);
			double[] normalizedPixels = normalize(pixels);
			for(int a = 0; a < normalizedPixels.length; a++) {
				normalizedPixels[a] = round(normalizedPixels[a], 2);
			}
			/*
			for(int a = 0; a < normalizedPixels.length; a++) {
				if(a % 32 == 0) {
					System.out.println();
				}
				System.out.printf("%.1f ", normalizedPixels[a]);
			}
			*/
			imgArr[imgP++] = normalizedPixels;
		}
		//print('\n');
		//print('\n');
		//print(Arrays.deepToString(imgArr));
		
		SigmoidLayer in = new SigmoidLayer(32 * 32, 2);
		ReLULayer out = new ReLULayer(2, 2);
		for(int a = 0; a < in.size(); a++) {
			in.setNode(a, new Node(2));
		}
		for(int a = 0; a < out.size(); a++) {
			out.setNode(a, new Node(2));
		}
		
		NeuralNetwork network = new NeuralNetwork(in, out);
		print(Arrays.deepToString(network.getAllOutputs(imgArr[0])));
		//network.getNewWeight(layerNumber, node, edge, input, expected, rate);
		double[] given = {2,2};
		double[] expected = {0,1};
		network.getNewWeight(1, 0, 0, imgArr[0], expected, 0.001);
		//for(int a = 0; a < 2;) {
		//	
		//}
	}
}


/* Paste in main
SigmoidLayer layer = new SigmoidLayer(5, 3);
SigmoidLayer layer2 = new SigmoidLayer(3, 3);

for(int a = 0; a < layer.size(); a++) {
	layer.setNode(a, new Node(3));
}

for(int a = 0; a < layer2.size(); a++) {
	layer2.setNode(a, new Node(3));
}

NeuralNetwork net = new NeuralNetwork(layer, layer2);

double[] input = {1,2,3,4,5};
double[] output = net.getOutput(input);
double[] expected = {0.9525740853634935, 0.9525740853634935, 0.9525740853634935};

println(Arrays.toString(output) + "\n");
println(net.getNewWeight(1, 0, 0, input, expected, 0.1));
*/
