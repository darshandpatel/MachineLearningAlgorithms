package ml;

import java.util.ArrayList;
import java.util.HashMap;

import Jama.Matrix;

public class Perceptron {

	HashMap<String,Object> dataMatrix;
	FileOperations fileOperations;
	
	public Perceptron(){
		fileOperations = new FileOperations();
		dataMatrix = fileOperations.fetchDataPointsFromFile(Constant.PERCEPTRON_DATA_FILE_PATH, 
				Constant.PERCEPTRON_DATA_FILE_NAME, Constant.PERCEPTRON_DATA_NUM_OF_DP,
				Constant.PERCEPTRON_NO_OF_FEATURES, Constant.STRING_REGEX);
	}
	
	public Matrix normalizeUpdateAttribute(Matrix attributeMatrix,
			Matrix targetMatrix,
			HashMap<Integer,HashMap<String,Double>> featureMinMaxMap){
		
		
		int noOfRows = attributeMatrix.getRowDimension();
		int noOfColumns = attributeMatrix.getColumnDimension();
		double[][] updatedttributeValues = new double[noOfRows][noOfColumns];
		
		for(int i = 0 ; i < noOfRows ; i++ ){
			
			updatedttributeValues[i][0] = 1;
			Double targetValue = targetMatrix.get(i, 0);
			for(int j = 1; j < noOfColumns; j++){
				
				Double multiplier = 1d;
				
				if(targetValue == -1d){
					multiplier = -1d;
				}
				
				HashMap<String,Double> minMaxMap = featureMinMaxMap.get(j);
				
				updatedttributeValues[i][j] = (multiplier) *((attributeMatrix.get(i, j) - 
						minMaxMap.get(Constant.MIN)) / 
						(minMaxMap.get(Constant.MAX) - minMaxMap.get(Constant.MIN)));
				
			}
			
		}
		return new Matrix(updatedttributeValues);
	}
	
	Matrix generateRandomCoefficientMatrix(){
		
		double coefficient[][] = new double[Constant.PERCEPTRON_NO_OF_FEATURES+1][1];
		
		for(int i=0;i<=Constant.PERCEPTRON_NO_OF_FEATURES;i++){
			coefficient[i][0] = Math.random();
		}
		return new Matrix(coefficient);
	}
	
	public void formPerceptron(){
		
		Matrix attributeMatrix = (Matrix) dataMatrix.get(Constant.ATTRIBUTES);
		Matrix targetMatrix = (Matrix) dataMatrix.get(Constant.TARGET);
		HashMap<Integer,HashMap<String,Double>> featureMinMaxMap = 
				(HashMap<Integer,HashMap<String,Double>>) dataMatrix.get(Constant.MIN_MAX_MAP);
		
		
		System.out.println("Attribute Matrix Dimension");
		System.out.println("Row : "+attributeMatrix.getRowDimension());
		System.out.println("Column : "+attributeMatrix.getColumnDimension());
		
		System.out.println("Target Matrix Dimension");
		System.out.println("Row : "+targetMatrix.getRowDimension());
		System.out.println("Column : "+targetMatrix.getColumnDimension());
		
		Matrix updatedAttributeMatrix = normalizeUpdateAttribute(attributeMatrix,
				targetMatrix,featureMinMaxMap);
		
		System.out.println("After updated Attribute Matrix Dimension");
		System.out.println("Row : "+updatedAttributeMatrix.getRowDimension());
		System.out.println("Column : "+updatedAttributeMatrix.getColumnDimension());
		
		Matrix coefficientMatrix = generateRandomCoefficientMatrix();
		
		System.out.println("Coefficient Matrix Dimension");
		System.out.println("Row : "+coefficientMatrix.getRowDimension());
		System.out.println("Column : "+coefficientMatrix.getColumnDimension());
		
		int iterationCount = 1;
		
		while(true){
			
			System.out.println("Iteration Count : "+iterationCount);
			
			ArrayList<Integer> negativeValueDP = new ArrayList<Integer>();
			
			
			// Run on every available data point
			for(int n = 0 ; n < Constant.PERCEPTRON_DATA_NUM_OF_DP ; n++){
				
				//System.out.println("Data Point number "+ n);
				
				int row[] = {n};
				Matrix rowMatrix = updatedAttributeMatrix.getMatrix(row,
						0,Constant.PERCEPTRON_NO_OF_FEATURES);
				
				//System.out.println("Row Matrix. # of rows " + rowMatrix.getRowDimension()
				//		+ " # of columns "+ rowMatrix.getColumnDimension());
				
				Matrix result = rowMatrix.times(coefficientMatrix);
				
				//System.out.println("Result Matrix dimension # of rows "+result.getRowDimension()
				//		+" # of columns "+result.getColumnDimension());
				
				if(result.get(0,0) < 0d){
					negativeValueDP.add(n);
					
					// Stochastic Update
					coefficientMatrix.plus(rowMatrix.timesEquals
							(Constant.LEARNING_RATE).transpose());
					
				}
			}
			System.out.println("Total Mistake : " + negativeValueDP.size());
			if(negativeValueDP.size() == 0){
				break;
			}
			
			System.out.println("Coefficient Matrix Dimension");
			System.out.println("Row : "+coefficientMatrix.getRowDimension());
			System.out.println("Column : "+coefficientMatrix.getColumnDimension());
			
			break;
		}
		
	}
}
