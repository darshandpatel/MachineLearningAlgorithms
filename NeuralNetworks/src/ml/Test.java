package ml;

import Jama.Matrix;

public class Test {
	
	public static void main(String args[]){
		
		double data[][] = {{1d,2d,3d},{4d,5d,6d},{7d,8d,9d}};
		Matrix dataMatrix = new Matrix(data);
		
		System.out.println(dataMatrix.getRowDimension());
		System.out.println(dataMatrix.getColumnDimension());
		
		int rows[] = {1,2};
		int columns[] = {1};
		Matrix subMatrix = dataMatrix.getMatrix(rows,columns);
		System.out.println(subMatrix.getRowDimension());
		System.out.println(subMatrix.getColumnDimension());
	}
}
