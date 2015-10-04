package ml;

import java.util.ArrayList;
import java.util.HashMap;
import Jama.Matrix;

public class Perceptron {

	HashMap<String,Object> dataMatrix;
	FileOperations fileOperations;
	
	/**
	 * Perceptron Constructor
	 * 
	 * This constructor fetches the Attribute and Matrix data from the source file
	 * 
	 */
	public Perceptron(){
		fileOperations = new FileOperations();
		dataMatrix = fileOperations.fetchDataPointsFromFile(Constant.PERCEPTRON_DATA_FILE_PATH, 
				Constant.PERCEPTRON_DATA_FILE_NAME, Constant.PERCEPTRON_DATA_NUM_OF_DP,
				Constant.PERCEPTRON_NO_OF_FEATURES, Constant.STRING_REGEX);
	}
	
	
	/**
	 * This method creates the Perceptron model of the given data.
	 */
	public void formPerceptron(){
		
		Matrix attributeMatrix = (Matrix) dataMatrix.get(Constant.ATTRIBUTES);
		Matrix targetMatrix = (Matrix) dataMatrix.get(Constant.TARGET);
		HashMap<Integer,HashMap<String,Double>> featureMinMaxMap = 
				(HashMap<Integer,HashMap<String,Double>>) dataMatrix.get(Constant.MIN_MAX_MAP);
		
		updateAttributeTargetMatrix(dataMatrix);
		
		//printMatrix(updatedAttributeMatrix);
		
		Matrix coefficientMatrix = generateRandomCoefficientMatrix();
		
		int iterationCount = 1;
		
		while(true){
			
			System.out.println("Iteration Count : "+iterationCount);
			
			ArrayList<Integer> negativeValueDP = new ArrayList<Integer>();
			
			// Loop on every available data point
			for(int n = 0 ; n < Constant.PERCEPTRON_DATA_NUM_OF_DP ; n++){
				
				//System.out.println("Data Point number "+ n);
				
				int row[] = {n};
				Matrix rowMatrix = attributeMatrix.getMatrix(row,
						0,Constant.PERCEPTRON_NO_OF_FEATURES);
				
				//printMatrix(rowMatrix);
				
				Matrix result = rowMatrix.times(coefficientMatrix);
				
				//printMatrix(result);
				
				if(result.get(0,0) < 0d){
					
					negativeValueDP.add(n);
					// Stochastic Update
					coefficientMatrix = coefficientMatrix.plus(rowMatrix.transpose().timesEquals
							(Constant.LEARNING_RATE));
				}
			}
			
			System.out.println("Total Mistake : " + negativeValueDP.size());
			if(negativeValueDP.size() == 0){
				break;
			}
			
			iterationCount++;
		}
		
		System.out.println("Feature Coefficient");
		
		printMatrix(coefficientMatrix.transpose());
		
	}
	
	/**
	 * This methods normalize the attribute value and modifies the sign of attribute 
	 * and target matrix data value if the target value is -1.0 
	 * 
	 * @param dataMatrix : The HashMap which contains the attribute, target and attribute
	 * Min Max value Objects.
	 * 
	 */
	private void updateAttributeTargetMatrix(HashMap<String,Object> dataMatrix){
		
		Matrix attributeMatrix = (Matrix) dataMatrix.get(Constant.ATTRIBUTES);
		Matrix targetMatrix = (Matrix) dataMatrix.get(Constant.TARGET);
		HashMap<Integer,HashMap<String,Double>> featureMinMaxMap = 
				(HashMap<Integer,HashMap<String,Double>> )dataMatrix.get(Constant.MIN_MAX_MAP);
		
		int noOfRows = attributeMatrix.getRowDimension();
		int noOfColumns = attributeMatrix.getColumnDimension();
		// Normalization method : Rescaling 
		for(int i = 0 ; i < noOfRows ; i++ ){
			
			Double targetValue = targetMatrix.get(i, 0);
			
			Double multiplier = 1d;
			
			if(targetValue == -1d){
				
				multiplier = -1d;
				targetMatrix.set(i, 0, 1d);
				
				/*
				attributeMatrix.set(i,0, -1d);
				for(int j = 1; j < noOfColumns; j++){
					attributeMatrix.set(i,j, (multiplier * (attributeMatrix.get(i, j))));
				}
				*/
			}
			
			attributeMatrix.set(i,0, multiplier);
			
			for(int j = 1; j < noOfColumns; j++){
				
				HashMap<String,Double> minMaxMap = featureMinMaxMap.get(j);
				attributeMatrix.set(i,j, (multiplier * ((attributeMatrix.get(i, j) - 
						minMaxMap.get(Constant.MIN)) / 
						(minMaxMap.get(Constant.MAX) - minMaxMap.get(Constant.MIN)))));
			}
			
		}
		
	}
	
	/**
	 * This method generates the random feature coefficient matrix
	 * @return The feature coefficient matrix
	 */
	private Matrix generateRandomCoefficientMatrix(){
		
		double coefficient[][] = new double[Constant.PERCEPTRON_NO_OF_FEATURES+1][1];
		
		for(int i=0;i<=Constant.PERCEPTRON_NO_OF_FEATURES;i++){
			//coefficient[i][0] = 0d;
			coefficient[i][0] = Math.random();
		}
		return new Matrix(coefficient);
	}
	
	/**
	 * This method print the given Matrix
	 * @param matrix : Data Matrix
	 */
	private void printMatrix(Matrix matrix){
		
		int noOfRows = matrix.getRowDimension();
		int noOfColumns = matrix.getColumnDimension();
		
		for(int i=0; i<noOfRows ; i++){
			for(int j=0;j<noOfColumns;j++){
				System.out.print(matrix.get(i, j)+" ");
			}
			System.out.print("\n");
		}
	}

}
