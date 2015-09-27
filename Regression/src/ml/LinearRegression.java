package ml;

import java.util.HashMap;

import Jama.Matrix;

public class LinearRegression {
	
	FileOperations fileOperations;
	
	public LinearRegression(){
		fileOperations = new FileOperations();
	}
	
	/**
	 * This method forms and validate the Linear Regression model
	 * based upon the Housing Data Set.
	 */
	public void formHousingLinearRegression(){
		
		System.out.println("Generate Linear Regression By Train Data");
		System.out.println("Get Train Data");
		HashMap<String,Matrix> trainMatrixHashMap = fileOperations.fetchDataPointsFromFile(
				Constant.HOUSING_DATA_FILE_PATH,Constant.HOUSING_TRAINDATA_FILE,
				Constant.HOUSING_DATA_NUM_OF_TRAINING_DP, 
				Constant.HOUSING_DATA_NO_OF_FEATURES, Constant.STRING_REGEX);
		
		Matrix coefficientMatrix = formLinearRegression(trainMatrixHashMap.get(Constant.ATTRIBUTES),
				trainMatrixHashMap.get(Constant.TARGET));
		
		System.out.println("Get Test Data");
		HashMap<String,Matrix> testMatrixHashMap = fileOperations.fetchDataPointsFromFile(
				Constant.HOUSING_DATA_FILE_PATH,Constant.HOUSING_TESTDATA_FILE,
				Constant.HOUSING_DATA_NUM_OF_TESTING_DP, 
				Constant.HOUSING_DATA_NO_OF_FEATURES, Constant.STRING_REGEX);
		
		System.out.println("Validate Linear Regression By Test Data");
		validateLinearRegression(coefficientMatrix, testMatrixHashMap.get(Constant.ATTRIBUTES),
				testMatrixHashMap.get(Constant.TARGET));
		
	}
	
	/**
	 * 
	 * @param attributeMatrix 	: Matrix of attribute values
	 * @param targetMatrix		: Matrix of target values
	 * @return linear regression Coefficient matrix
	 */
	public Matrix formLinearRegression(Matrix attributeMatrix,Matrix targetMatrix){
		
		Matrix attributeMatrixTranspose = attributeMatrix.transpose();
		
		return attributeMatrixTranspose.times(attributeMatrix).inverse().
				times(attributeMatrixTranspose).times(targetMatrix);
		
	}
	
	/**
	 * 
	 * @param coefficientMatrix	: Linear Regression Coefficient matrix
	 * @param attributeMatrix	: Matrix of attribute values
	 * @param targetMatrix		: Matrix of target values
	 */
	public void validateLinearRegression(Matrix coefficientMatrix, Matrix attributeMatrix, 
			Matrix targetMatrix){
		
		Matrix predictedMatrix = attributeMatrix.times(coefficientMatrix);
		int numOfRows = predictedMatrix.getRowDimension();
		System.out.println("Total test records " + numOfRows);
		for(int n = 0 ; n < numOfRows ; n++){
			System.out.printf("%s : %f, %s : %f\n","Predicted Value",
					predictedMatrix.get(n, 0),"Actual Value",targetMatrix.get(n,0));
		}
		
		Matrix errorMatrix = predictedMatrix.minus(targetMatrix);
		
		double totalError = 0d;
		for(int n = 0 ; n < numOfRows ; n++){
			double value = errorMatrix.get(n, 0);
			totalError += Math.pow(value,2);
		}
		
		System.out.println("Mean square error is : "+ (totalError/numOfRows));
	}

}
