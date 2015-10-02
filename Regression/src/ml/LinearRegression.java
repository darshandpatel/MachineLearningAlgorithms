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
	
	
	public Double validateLinearRegression(Matrix coefficientMatrix, Matrix attributeMatrix, 
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
		Double meanSquareError = (totalError/numOfRows);
		System.out.println("Mean square error is : "+ meanSquareError);
		return meanSquareError;
	}
	
	/** 
	 * This method forms and validates the linear regression model
	 * based upon the Spam database.
	 */
	public void formMultipleSpamLinearRegression(){
		
		HashMap<String,Matrix> matrixHashMap  =  fileOperations.fetchDataPointsFromFile(
				Constant.SPAMBASE_DATA_FILE_PATH,Constant.SPAMBASE_DATA_FILE_NAME,
				Constant.SPAMBASE_DATA_NUM_OF_DP,Constant.SPAMBASE_DATA_NUM_OF_FEATURES + 1
				,Constant.COMMA_REGEX);
		
		// Number of fold will be created on the given Data Matrix;
			int numOfFolds = 10;
			int currentFoldCount = 1;
			
			// Total average accuracy of all the Decision Trees.
			Double sumOfMSE = 0d;
			
			while(currentFoldCount <= numOfFolds){
				
				//System.out.println("Current Fold number" + currentFoldCount);
				
				// Form Decision Tree
				System.out.println("===============================");
				System.out.println("Fold count : "+currentFoldCount);
				
				HashMap<String,HashMap<String,Matrix>> fullDataMatrix = 
						formDataMatrixByFold(matrixHashMap,numOfFolds,currentFoldCount);
				
				HashMap<String,Matrix> trainMatrixHashMap = fullDataMatrix.get(Constant.TRAIN);
				HashMap<String,Matrix> testMatrixHashMap = fullDataMatrix.get(Constant.TEST);
				
				
				Matrix coefficientMatrix = formLinearRegression(
						trainMatrixHashMap.get(Constant.ATTRIBUTES),
						trainMatrixHashMap.get(Constant.TARGET));
				
				System.out.println("Validate Linear Regression By Test Data");
				Double meanSquareError = validateLinearRegression(coefficientMatrix, 
							testMatrixHashMap.get(Constant.ATTRIBUTES),
							testMatrixHashMap.get(Constant.TARGET));
				
				
				// Accumulate Results
				sumOfMSE += meanSquareError;
				//break;
				currentFoldCount++;
				
			}
			
			// Print the Results
			System.out.println("Final accuracy : "+ sumOfMSE/numOfFolds);
				
		
	}
	
	public HashMap<String,HashMap<String,Matrix>> formDataMatrixByFold(
			HashMap<String,Matrix> dataMatrixHashMap,
			Integer numOfFolds,Integer currentFoldCount){
		
		Matrix attributeMatrix = dataMatrixHashMap.get(Constant.ATTRIBUTES);
		Matrix targetMatrix = dataMatrixHashMap.get(Constant.TARGET);
		
		HashMap<String,HashMap<String,Matrix>> fullMatrixMap 
			= new HashMap<String,HashMap<String,Matrix>>();
		
		HashMap<String,Matrix> trainMatrixHashMap = new HashMap<String,Matrix>();
		HashMap<String,Matrix> testMatrixHashMap = new HashMap<String,Matrix>();
		
		int numOfDPPerFold = Constant.SPAMBASE_DATA_NUM_OF_DP / numOfFolds;
		
		// Extra Data Points
		int trainExtraDP = (currentFoldCount != numOfFolds?Constant.SPAMBASE_DATA_NUM_OF_DP % numOfFolds:0);
		int testExtraDP = (currentFoldCount == numOfFolds?Constant.SPAMBASE_DATA_NUM_OF_DP % numOfFolds:0);
		
		int numOfDPInTrain = ((numOfFolds - 1) * numOfDPPerFold) + trainExtraDP;
		int numOfDPInTest = numOfDPPerFold + testExtraDP;
		
		double trainAttributeData[][] = 
				new double[numOfDPInTrain][Constant.SPAMBASE_DATA_NUM_OF_FEATURES+1];
		double testAttributeData[][] = 
				new double[numOfDPInTest][Constant.SPAMBASE_DATA_NUM_OF_FEATURES+1];
		double trainTargetData[][] = 
				new double[numOfDPInTrain][1];
		double testTargetData[][] = 
				new double[numOfDPInTest][1];
		
		int trainDPIndex = 0;
		int testDPIndex = 0;
		
		for(int i = 1; i <= numOfFolds; i++){
			
			int startIndex = (i - 1) * numOfDPPerFold;
			
			if(i != currentFoldCount){
				int endIndex = startIndex + (numOfDPPerFold - 1) + 
						(currentFoldCount == numOfFolds?trainExtraDP:0);
				while(startIndex <= endIndex){
					for(int j = 0 ; j <= Constant.SPAMBASE_DATA_NUM_OF_FEATURES;j++)
						trainAttributeData[trainDPIndex][j] = attributeMatrix.get(startIndex, j);
					trainTargetData[trainDPIndex][0] = targetMatrix.get(startIndex, 0);
					startIndex++;
					trainDPIndex++;
				}
			}else{
				int endIndex = startIndex + (numOfDPPerFold - 1) + 
						(currentFoldCount == numOfFolds?testExtraDP:0);
				while(startIndex <= endIndex){
					for(int j = 0 ; j <= Constant.SPAMBASE_DATA_NUM_OF_FEATURES;j++)
						testAttributeData[testDPIndex][j] = attributeMatrix.get(startIndex, j);
					testTargetData[testDPIndex][0] = targetMatrix.get(startIndex, 0);
					startIndex++;
					testDPIndex++;
				}
				
			}
		}
		
		trainMatrixHashMap.put(Constant.ATTRIBUTES, new Matrix(trainAttributeData));
		trainMatrixHashMap.put(Constant.TARGET, new Matrix(trainTargetData));
		
		testMatrixHashMap.put(Constant.ATTRIBUTES, new Matrix(testAttributeData));
		testMatrixHashMap.put(Constant.TARGET, new Matrix(testTargetData));
		
		fullMatrixMap.put(Constant.TRAIN,trainMatrixHashMap);
		fullMatrixMap.put(Constant.TEST, testMatrixHashMap);
		
		return fullMatrixMap;
	}
	

}
