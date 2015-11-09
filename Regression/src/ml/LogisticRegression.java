package ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.google.common.primitives.Doubles;

import Jama.Matrix;

public class LogisticRegression {
	
	private FileOperations fileOperations;
	private HashMap<Integer,List<Integer>> dataPointPerFold;
	
	HashMap<String,Matrix> matrixHashMap;
	
	public LogisticRegression(){
		fileOperations = new FileOperations();
		matrixHashMap  =  fileOperations.fetchDataPointsFromFile(
				Constant.SPAMBASE_DATA_FILE_PATH,Constant.SPAMBASE_DATA_FILE_NAME,
				Constant.SPAMBASE_DATA_NUM_OF_DP,Constant.SPAMBASE_DATA_NUM_OF_FEATURES + 1
				,Constant.COMMA_REGEX);
	}
	
	public HashMap<Integer,List<Integer>> formDifferentDataPerFold(){
		
		dataPointPerFold = new HashMap<Integer,List<Integer>>();
		
		int numOfDPPerFold = Constant.SPAMBASE_DATA_NUM_OF_DP / Constant.NBR_OF_FOLDS;
		int extraDP = Constant.SPAMBASE_DATA_NUM_OF_DP % Constant.NBR_OF_FOLDS;
		
		for(int i = 0; i < Constant.NBR_OF_FOLDS; i++){
			
			List<Integer> numbers = IntStream.iterate(i, n -> n + 10).limit(numOfDPPerFold + ((i ==0)?extraDP:0))
					.boxed().collect(Collectors.toList());
			dataPointPerFold.put(i, numbers);
		}
		
		return dataPointPerFold;
		
	}

	
	public Matrix formLogisticRegression(Matrix attributeMatrix,Matrix targetMatrix){
	
		Integer nbrOfDp = attributeMatrix.getRowDimension();
		
		Matrix weightMatrix = generateRandomWeight();
		Double previoudLogLikelihood = 0d;
		do{
			
			
			for(int i = 0; i < nbrOfDp; i++){
				
				previoudLogLikelihood = calculateLogLikelihood(attributeMatrix, 
						targetMatrix, weightMatrix);
				
				int[] row = {i};
				
				Matrix attributeValue = attributeMatrix.getMatrix(row, 0, 
						Constant.SPAMBASE_DATA_NUM_OF_FEATURES);
				Double actualValue = targetMatrix.get(i, 0);
				
				double predictedValue = predictionValue(attributeValue, weightMatrix);
				
				weightMatrix = updateWeight(predictedValue, actualValue, 
						attributeValue, weightMatrix);
				
				Double newLogLikelihood = calculateLogLikelihood(attributeMatrix, 
						targetMatrix, weightMatrix);
				Double diff = Math.abs(previoudLogLikelihood - newLogLikelihood);
				System.out.println(diff);
				if(diff < Constant.THRESHOLD_VALUE){
					return weightMatrix;
				}
				
			}
			
		}while(true);
		
	}
	
	public Matrix generateRandomWeight(){
		return Matrix.random(Constant.SPAMBASE_DATA_NUM_OF_FEATURES + 1, 1);
	}
	
	public Matrix updateWeight(Double predictedValue, Double actualValue, Matrix attributeValue,
			Matrix weightMatrix){
		
		Double multiplier = Constant.LEARNING_RATE * ( predictedValue - actualValue);
		double[][] updatedWeights = new double[Constant.SPAMBASE_DATA_NUM_OF_FEATURES+1][1]; 
		
		for(int i = 0; i <= Constant.SPAMBASE_DATA_NUM_OF_FEATURES; i++){
			
			updatedWeights[i][0] = weightMatrix.get(i, 0) - (multiplier * attributeValue.get(0, i)); 
			
		}
		
		return new Matrix(updatedWeights);
		
	}
	
	public Double calculateLogLikelihood(Matrix attributeMatrix,Matrix targetMatrix,
			Matrix weightMatrix){
		
		Double logLikelihood = 0d;
		int nbrOfDp = attributeMatrix.getRowDimension();
		
		for(int i = 0; i < nbrOfDp; i++){
			
			int[] row = {i};
			
			Matrix attributeValue = attributeMatrix.getMatrix(row, 0, 
					Constant.SPAMBASE_DATA_NUM_OF_FEATURES);
			Double actualValue = targetMatrix.get(i, 0);
			
			double predictedValue = predictionValue(attributeValue, weightMatrix);
			
			if(actualValue == 1){
				logLikelihood += Math.log(predictedValue);
			}else{
				logLikelihood += Math.log(1 - predictedValue);
			}
			
		}
		return logLikelihood;
		
	}
	
	public Double predictionValue(Matrix attributeValue, Matrix weightMatrix){
		
		double sum = attributeValue.times(weightMatrix).get(0, 0);
		return (1d / (1d + Math.pow(Math.E, -sum)));
		
	}
	
	
	public Double validateLogisticRegression(Matrix coefficientMatrix, Matrix attributeMatrix, 
			Matrix targetMatrix){
		
		int nbrOfTestDP = attributeMatrix.getRowDimension();
		int nbrOfCorrectlyPredictedDP = 0;
		for(int i = 0; i < nbrOfTestDP; i++){
			
			int[] row = {i};
			Matrix attributeValues = attributeMatrix.getMatrix(row, 0, 
					Constant.SPAMBASE_DATA_NUM_OF_FEATURES);
			Double predictedValue = predictionValue(attributeValues, coefficientMatrix);
			System.out.println("Predicted Value " + predictedValue);
			
			if(predictedValue > 0.5){
				
				if(targetMatrix.get(1,0) == 1){
					nbrOfCorrectlyPredictedDP++;
				}
				
			}else{
				
				if(targetMatrix.get(1,0) == 0){
					nbrOfCorrectlyPredictedDP++;
				}
				
			}
			
		}
		
		double accuracy = (double)nbrOfCorrectlyPredictedDP/nbrOfTestDP;
		System.out.println("Accuracy : "+ accuracy);
		return accuracy;
		
	}
	
	/** 
	 * This method forms and validates the Logistic regression model
	 * based upon the Spam database.
	 */
	public void formSpamLogisticRegression(){
		
		// Number of fold will be created on the given Data Matrix;
		int numOfFolds = 10;
		int currentFoldCount = 1;
		
		// Total average accuracy of all the Decision Trees.
		Double sumOfAccuracy = 0d;
		formDifferentDataPerFold();
		
		while(currentFoldCount <= numOfFolds){
			
			//System.out.println("Current Fold number" + currentFoldCount);
			
			// Form Decision Tree
			System.out.println("Fold count : " + currentFoldCount);
			
			HashMap<String,HashMap<String,Matrix>> fullDataMatrix = 
					formDataMatrixByFold(matrixHashMap,numOfFolds,currentFoldCount);
			
			HashMap<String,Matrix> trainMatrixHashMap = fullDataMatrix.get(Constant.TRAIN);
			HashMap<String,Matrix> testMatrixHashMap = fullDataMatrix.get(Constant.TEST);
			
			
			Matrix coefficientMatrix = formLogisticRegression(
					trainMatrixHashMap.get(Constant.ATTRIBUTES),
					trainMatrixHashMap.get(Constant.TARGET));
			
			System.out.println("Validate Linear Regression By Test Data");
			Double accuracy = validateLogisticRegression(coefficientMatrix, 
						testMatrixHashMap.get(Constant.ATTRIBUTES),
						testMatrixHashMap.get(Constant.TARGET));
			currentFoldCount++;
			sumOfAccuracy += accuracy;
			break;
			
		}
		
		// Print the Results
		System.out.println("Final accuracy : "+ sumOfAccuracy/numOfFolds);
		
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
		
		ArrayList<ArrayList<Double>> trainAttributes = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> testAttributes = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> trainTargets = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> testTargets = new ArrayList<ArrayList<Double>>();
		
		ArrayList<Integer> testDPList = new ArrayList<Integer>();
		ArrayList<Integer> trainDPList = new ArrayList<Integer>();
		
		for(int i = 0; i < Constant.NBR_OF_FOLDS; i++){
			
			if(i != currentFoldCount){
				
				List<Integer> dataPoints = dataPointPerFold.get(i);
				Iterator<Integer> dataPointIterator = dataPoints.iterator();
				
				while(dataPointIterator.hasNext()){
					
					int rowCount = dataPointIterator.next();
					ArrayList<Double> attribeValues = new ArrayList<Double>();
					ArrayList<Double> targetValue = new ArrayList<Double>();
					for(int j = 0 ; j <= Constant.SPAMBASE_DATA_NUM_OF_FEATURES;j++)
						attribeValues.add(attributeMatrix.get(rowCount, j));
					targetValue.add(targetMatrix.get(rowCount, 0));
					trainAttributes.add(attribeValues);
					trainTargets.add(targetValue);
					trainDPList.add(rowCount);
				}
				
			}else{
				
				List<Integer> dataPoints = dataPointPerFold.get(i);
				Iterator<Integer> dataPointIterator = dataPoints.iterator();
				
				while(dataPointIterator.hasNext()){
					
					int rowCount = dataPointIterator.next();
					ArrayList<Double> attribeValues = new ArrayList<Double>();
					ArrayList<Double> targetValue = new ArrayList<Double>();
					for(int j = 0 ; j <= Constant.SPAMBASE_DATA_NUM_OF_FEATURES;j++)
						attribeValues.add(attributeMatrix.get(rowCount, j));
					targetValue.add(targetMatrix.get(rowCount, 0));
					testAttributes.add(attribeValues);
					testTargets.add(targetValue);
					testDPList.add(rowCount);
				}
					
			}
			
		}
		
		// Convert the ArrayList into 2 D array
		double trainAttributeData[][] = new double[trainAttributes.size()][];
		double testAttributeData[][] = new double[testAttributes.size()][];
		
		double trainTargetData[][] = new double[trainTargets.size()][];
		double testTargetData[][] = new double[testTargets.size()][];
		
		int nbrOFTrainData = trainAttributes.size();
		int nbrOfTestData = testAttributes.size();
		
		for (int i = 0; i < nbrOFTrainData; i++) {
		    ArrayList<Double> row = trainAttributes.get(i);
		    trainAttributeData[i] = Doubles.toArray(row);
		}
		
		for (int i = 0; i < nbrOFTrainData; i++) {
		    ArrayList<Double> row = trainTargets.get(i);
		    trainTargetData[i] = Doubles.toArray(row);
		}
		
		for (int i = 0; i < nbrOfTestData; i++) {
		    ArrayList<Double> row = testAttributes.get(i);
		    testAttributeData[i] = Doubles.toArray(row);
		}
		
		for (int i = 0; i < nbrOfTestData; i++) {
		    ArrayList<Double> row = testTargets.get(i);
		    testTargetData[i] = Doubles.toArray(row);
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
