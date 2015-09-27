package ml;

import Jama.Matrix;

public class MatrixTest {
	
	public static void main(String args[]){
		
		double data[][] = {{1,1},{4,5}};
		double newData[][] = {{1,1},{4,5}};
		
		Matrix dataMatrix = new Matrix(data);
		System.out.println(dataMatrix.getRowDimension());
		System.out.println(dataMatrix.getColumnDimension());
		
		Matrix transposeMatrix = dataMatrix.transpose();
		System.out.println(transposeMatrix.get(1, 0));
		System.out.println(dataMatrix.times(transposeMatrix).get(1, 0));
		
		String abc = "0.00632  18.00   2.310  0  0.5380  6.5750  65.20  4.0900   1  296.0  15.30 396.90   4.98  24.00";
		String parts[] = abc.split("\\s+");
		System.out.println("Length of parts : " + parts.length);
	}
	

}
