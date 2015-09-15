package ml;

import java.util.LinkedList;
import java.util.Queue;

public class Test {
	
	public static void main(String args[]){
		
		LinkedList<Feature> selectedFeature = new LinkedList<Feature>();
		//selectedFeature.get(index)
		//Queue<E>
		String abc = "0.00632  18.00   2.310  0  0.5380  6.5750  65.20  4.0900   1  296.0  15.30 396.90   4.98  24.00";
		String parts[] = abc.split("\\s+");
		System.out.println("Split parts : "+ parts.length);
		for(int i = 0; i < parts.length ; i++){
			System.out.println(parts[i]);
		}
	}

}
