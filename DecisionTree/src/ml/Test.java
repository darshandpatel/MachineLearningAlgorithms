package ml;

import java.util.LinkedList;
import java.util.Queue;

import Jama.Matrix;

public class Test {
	
	public static void main(String args[]){
		
		double array[][] = {{1,2,3},{4,5,6}};
		Matrix matrix = new Matrix(array);
		
		System.out.println(matrix.getColumnDimension());
		System.out.println(matrix.getRowDimension());
		
		
	}

}
