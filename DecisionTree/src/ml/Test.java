package ml;

import java.util.LinkedList;
import java.util.Queue;

import Jama.Matrix;

public class Test {
	
	public static void main(String args[]){
		
		double array[][] = {{1,2,3},{4,5,6},{7,8,9}};
		Matrix matrix = new Matrix(array);
		
		System.out.println(matrix.getColumnDimension());
		System.out.println(matrix.getRowDimension());
		Matrix newMatrix = matrix.getMatrix(1, 2, new int[2]);
		
		for(int i=0;i<newMatrix.getRowDimension();i++){
			System.out.println(newMatrix.get(i,0));
			
		}
		
		
	}

}
