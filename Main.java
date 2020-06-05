import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;

public class Main {
	
	public static void println(Object a) {
		System.out.println(a);
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
		
		BufferedImage img = null;
		try {
			img = ImageIO.read(new File("one.jpg"));
		}
		catch(IOException e) {
			
		}
		int[] pixels = new int[32 * 32];
		
		//Extract pixels
		int pPointer = 0;
		for(int a = 0; a < 32; a++) {
			for(int b = 0; b < 32; b++) {
				pixels[pPointer++] = img.getRGB(a, b);
			}
		}
		invert(pixels);
		double[] normalizedPixels = normalize(pixels);
		for(int a = 0; a < normalizedPixels.length; a++) {
			normalizedPixels[a] = round(normalizedPixels[a], 2);
		}
		println(Arrays.toString(normalizedPixels));
		
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
