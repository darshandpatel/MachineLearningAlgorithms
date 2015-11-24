package ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

import com.google.common.primitives.Ints;

import Jama.Matrix;



public class AdaBoost {
	
	static HashMap<Integer,Double> trainDataDistribution;
	FileOperations fileOperations;
	
	public AdaBoost(){
		fileOperations = new FileOperations();
	}
	
	
	public void performAdaBoosting(){
		
		HashMap<Integer,String> attributeType = new HashMap<Integer,String>();
		for(int i=0;i<Constant.NBR_OF_FEATURES;i++){
			attributeType.put(i, Constant.NUMERIC);
		}
		
		Matrix dataMatrix = fileOperations.fetchDataPointsFromFile(
				Constant.SPAMBASE_DATA_FILE_PATH,Constant.SPAMBASE_DATA_FILE_NAME,
				Constant.NBR_OF_DP,Constant.NBR_OF_FEATURES + 1
				,Constant.COMMA_REGEX);
		
		DecisionStamp decisionStamp = new DecisionStamp(dataMatrix);
		HashMap<Integer,List<Integer>> dataPointPerFold = decisionStamp.formDifferentDataPerFold(
				Constant.NBR_OF_DP, Constant.NBR_OF_FOLDS);
		
		// Number of fold will be created on the given Data Matrix;
		int numOfFolds = 10;
		int currentFoldCount = 0;
		
		// Total average accuracy of all the Decision Trees.
		Double sumOfAccuracy = 0d;
		
		while(currentFoldCount < numOfFolds){
			
			System.out.println("Fold count : " + currentFoldCount);
			
			HashMap<String,List<Integer>> matrixHashMap = decisionStamp.formTrainTestDPByFold(numOfFolds,currentFoldCount);
			
			List<Integer> trainDPList = (List<Integer>) matrixHashMap.get(Constant.TRAIN_DP);
			List<Integer> testDPList = (List<Integer>) matrixHashMap.get(Constant.TEST_DP);
			
			ArrayList<Feature> features = decisionStamp.fetchFeaturePosThreshold(trainDPList,attributeType);
			formInitialDataDistribution(dataPointPerFold, currentFoldCount);
			
			ArrayList<Double> trainErrors = new ArrayList<Double>();
			ArrayList<Node> rootNodes = new ArrayList<Node>();
			ArrayList<Double> alphas = new ArrayList<Double>();
			ArrayList<Double> weightedErrors = new ArrayList<Double>();
			int weakLearnerCount = 0;
			
			while(weakLearnerCount < 100){
				
				System.out.println("Weak Learner Count :" + weakLearnerCount);
				
				// Apply the weak learner on the training Data set.
				Node rootNode = new Node();
				rootNode.setDataPoints(trainDPList);

				decisionStamp.formTrainDecisionTree(rootNode, features, trainDataDistribution,Constant.SPAMBASE_DATA_DEPTH_LIMIT,
						Constant.BAL_DATA_TARGET_VALUE_INDEX);
				
				//Calculate the weighted error.
				// Evaluate Decision Tree
				HashMap<String, Object> returnMap = decisionStamp.calculateWeightedError(trainDPList, 
						rootNode, trainDataDistribution, Constant.SPAMBASE_DATA_TARGET_VALUE_INDEX);
				
				Double weighedError = (Double)returnMap.get(Constant.ERROR_VALUE);
				weightedErrors.add(weighedError);
				System.out.println("Weighted Error " + weighedError);
				// Calculate the alpha
				
				double alpha = (double)(1d/2d) * (Math.log((1 - weighedError) / weighedError));
				System.out.println("Alpha :" + alpha);
				returnMap.put(Constant.ALPHA_VALUE, alpha);

				// Update the data distribution
				updateDataDistribution(returnMap);	
				rootNodes.add(rootNode);
				alphas.add(alpha);
				trainErrors.add(validateTrainData(rootNodes, alphas, testDPList, dataMatrix));
				weakLearnerCount++;
			}
			// Print the Results
			validateTestData(rootNodes, alphas, testDPList, dataMatrix, Constant.SPAMBASE_DATA_TARGET_VALUE_INDEX);
			System.out.println("Train Error Data");
			System.out.println(trainErrors);
			System.out.println("Round Error per Iteration");
			System.out.println(weightedErrors);
			break;
		}
		

		

	}
	
	public static void validateTestData(ArrayList<Node> rootNodes, ArrayList<Double> alphas,
			List<Integer> testDPList, Matrix dataMatrix, Integer targetValueIndex){
		
		double dataArray[][] = dataMatrix.getArray();
		
		Integer numOfTestDP = testDPList.size();
		int count = rootNodes.size();
		int correctCount = 0;
		ArrayList<Double> errorsList = new ArrayList<Double>();
		
		for(int j = 0; j < count; j++){
			
			Double error = 0d;
			correctCount = 0;
			
			for(int i=0;i< numOfTestDP;i++){
				
				Double score = 0d;
				Integer testDP = testDPList.get(i);
				
				for(int k = 0 ; k <= j; k++){
					score += alphas.get(k) * 
							DecisionStamp.predictionValue(dataArray[testDP],rootNodes.get(k));
				}
				
				double actualTargetValue = dataArray[testDP][targetValueIndex];
				double predictedValue = 0d;
				
				if (score > 0)
					predictedValue = 1d;
				else
					predictedValue = -1d;
				
				if(actualTargetValue == predictedValue)
					correctCount++;
			}
			
			error = 1d - (double)correctCount/numOfTestDP;
			errorsList.add(error);
			
		}
		
		System.out.println("Prediction Accuracy :" + ((double)correctCount/numOfTestDP));
		System.out.println("Test Errors");
		System.out.println(errorsList);
		
	}
	
	public static void formInitialDataDistribution(HashMap<Integer,List<Integer>> dataPointPerFold, Integer currentDataFold){
		
		trainDataDistribution = new HashMap<Integer, Double>();
		
		Iterator<Map.Entry<Integer,List<Integer>>> iterator = dataPointPerFold.entrySet().iterator();
		int nbrOfTrainDP = 0;
		while(iterator.hasNext()){
			
			Entry<Integer,List<Integer>> pair = iterator.next();
			if(pair.getKey() != currentDataFold)
				nbrOfTrainDP += pair.getValue().size();
			
		}
		
		iterator = dataPointPerFold.entrySet().iterator();
		
		while(iterator.hasNext()){
			
			Entry<Integer,List<Integer>> pair = iterator.next();
			int turn = pair.getKey();
			List<Integer> dataPoints = pair.getValue();
			if(turn != currentDataFold){
				for(Integer dataPoint : dataPoints){
					trainDataDistribution.put(dataPoint, (double)(1.0d/nbrOfTrainDP));
				}
			}
			
		}
		
	}
	
	public static void formInitialDataDistribution(List<Integer> trainDPList){
		
		trainDataDistribution = new HashMap<Integer, Double>();
		
		int nbrOfTrainDP = trainDPList.size();
		Iterator<Integer> iterator = trainDPList.iterator();
		
		while(iterator.hasNext()){
			Integer dataPoint = iterator.next();
			trainDataDistribution.put(dataPoint, (double)(1.0d/nbrOfTrainDP));
		}
		
	}
	
	public static void updateDataDistribution(HashMap<String, Object> returnMap){
		
		
		HashMap<Integer,Double> oldDataDistribution = (HashMap<Integer, Double>) trainDataDistribution.clone();
		
		
		Double alpha = (Double) returnMap.get(Constant.ALPHA_VALUE);
		List<Integer> misClassifiedDP = (List<Integer>) returnMap.get(Constant.MISCLASSIFIED_DP);
		List<Integer> correctlyClassifiedDP = (List<Integer>) returnMap.get(Constant.CORRECTLY_CLASSIFIED_DP);
		
		Double updatedDataDistirbution;
		Double sumOfDistributionValue = 0d;
		
		//System.out.println("Miss classified data points");
		//System.out.println(misClassifiedDP);
		//System.out.println("Old value for the miss classified data");
		for(Integer dataPoint : misClassifiedDP){
			
			//System.out.print(dataPoint+":"+trainDataDistribution.get(dataPoint)+" ");
			updatedDataDistirbution = trainDataDistribution.get(dataPoint) * Math.pow(Math.E,alpha);
			sumOfDistributionValue += updatedDataDistirbution;
			trainDataDistribution.put(dataPoint, updatedDataDistirbution);
			
		}
		
		
		for(Integer dataPoint : correctlyClassifiedDP){
			
			updatedDataDistirbution = trainDataDistribution.get(dataPoint) * Math.pow(Math.E,-alpha);
			sumOfDistributionValue += updatedDataDistirbution;
			trainDataDistribution.put(dataPoint, updatedDataDistirbution);
			
		}
		
		// Normalize the Distribution
		for(Integer dataPoint : misClassifiedDP){
			trainDataDistribution.put(dataPoint, trainDataDistribution.get(dataPoint)/sumOfDistributionValue);
		}
		
		for(Integer dataPoint : correctlyClassifiedDP){
			trainDataDistribution.put(dataPoint, trainDataDistribution.get(dataPoint)/sumOfDistributionValue);
		}
		
		/*
		System.out.println("New value for the miss classified data");
		for(Integer dataPoint : misClassifiedDP){
			
			System.out.print(dataPoint+":"+trainDataDistribution.get(dataPoint)+" ");
		}
		*/
		
	}
	
	public HashMap<String,Object> performAdaboosting(Matrix dataMatrix, List<Integer> trainDPList,
			int nbrOfAllowedWeakLearner){
		
		int weakLearnerCount = 0;
		
		HashMap<String,Object> returnMap = new HashMap<String,Object>();
		
		HashMap<Integer,String> attributeType = new HashMap<Integer,String>();
		for(int i=0;i<Constant.NBR_OF_FEATURES;i++){
			attributeType.put(i, Constant.NUMERIC);
		}
		
		DecisionStamp decisionStamp = new DecisionStamp(dataMatrix);
		ArrayList<Feature> features = decisionStamp.fetchFeaturePosThreshold(trainDPList,attributeType);
		List<Node> rootNodes = new ArrayList<Node>();
		ArrayList<Double> alphas = new ArrayList<Double>();
		
		formInitialDataDistribution(trainDPList);
		
		while(weakLearnerCount < nbrOfAllowedWeakLearner){
			
			//System.out.println("Weak Learner Count :" + weakLearnerCount);
			
			// Apply the weak learner on the training Data set.
			Node rootNode = new Node();
			rootNode.setDataPoints(trainDPList);
			
			decisionStamp.formTrainDecisionTree(rootNode, features, trainDataDistribution,Constant.SPAMBASE_DATA_DEPTH_LIMIT,
					Constant.BAL_DATA_TARGET_VALUE_INDEX);

			//Calculate the weighted error.
			// Evaluate Decision Tree
			HashMap<String, Object> map = decisionStamp.calculateWeightedError(trainDPList, 
					rootNode, trainDataDistribution, Constant.SPAMBASE_DATA_TARGET_VALUE_INDEX);
			
			Double weighedError = (Double)map.get(Constant.ERROR_VALUE);
			//System.out.println("Weighted Error " + weighedError);
			// Calculate the alpha
			double alpha = (double)(1d/2d) * (Math.log((1 - weighedError) / weighedError));
			//System.out.println("Alpha :" + alpha);
			map.put(Constant.ALPHA_VALUE, alpha);

			// Update the data distribution
			updateDataDistribution(map);	
			rootNodes.add(rootNode);
			alphas.add(alpha);
			weakLearnerCount++;
		}
		
		returnMap.put(Constant.ROOT_NODES, rootNodes);
		returnMap.put(Constant.ALPHAS, alphas);
		return returnMap;
		
	}
	
	public Matrix mergetAttributeTargetMatrix(Matrix attributeMatrix, Matrix targetMatrix){
		
		int nbrOfDP = attributeMatrix.getRowDimension();
		int nbrOfColumn = attributeMatrix.getColumnDimension() + 1;
		
		double dataMatrix[][] = new double[nbrOfDP][nbrOfColumn];
		
		for(int i = 0; i < nbrOfDP ; i++){
			
			for(int j=0;j<nbrOfColumn-1;j++){
				dataMatrix[i][j] = attributeMatrix.get(i, j);
			}
			dataMatrix[i][nbrOfColumn-1] = targetMatrix.get(i, 0);
		}
		
		return new Matrix(dataMatrix);
		
	}
	public HashMap<String,Object> performAdaboosting(Matrix attributeMatrix, Matrix targetMatrix, List<Integer> trainDPList,
			HashMap<Integer,String> attributeType, int nbrOfallowedWeakLeaner, int treeDepth,int targetValueIndex){
		
		int weakLearnerCount = 0;
		
		Matrix dataMatrix = mergetAttributeTargetMatrix(attributeMatrix, targetMatrix);
		
		HashMap<String,Object> returnMap = new HashMap<String,Object>();
		
		DecisionStamp decisionStamp = new DecisionStamp(dataMatrix);
		ArrayList<Feature> features = decisionStamp.fetchFeaturePosThreshold(trainDPList,attributeType);
		List<Node> rootNodes = new ArrayList<Node>();
		ArrayList<Double> alphas = new ArrayList<Double>();
		
		formInitialDataDistribution(trainDPList);
		
		while(weakLearnerCount < nbrOfallowedWeakLeaner){
			
			//System.out.println("Weak Learner Count :" + weakLearnerCount);
			
			// Apply the weak learner on the training Data set.
			Node rootNode = new Node();
			rootNode.setDataPoints(trainDPList);
			
			decisionStamp.formTrainDecisionTree(rootNode, features, trainDataDistribution,treeDepth,
					targetValueIndex);

			//Calculate the weighted error.
			// Evaluate Decision Tree
			HashMap<String, Object> map = decisionStamp.calculateWeightedError(trainDPList, 
					rootNode, trainDataDistribution, targetValueIndex);
			
			Double weighedError = (Double)map.get(Constant.ERROR_VALUE);
			//System.out.println("Weighted Error " + weighedError);
			// Calculate the alpha
			double alpha = (double)(1d/2d) * (Math.log((1 - weighedError) / weighedError));
			//System.out.println("Alpha :" + alpha);
			map.put(Constant.ALPHA_VALUE, alpha);

			// Update the data distribution
			updateDataDistribution(map);	
			rootNodes.add(rootNode);
			alphas.add(alpha);
			weakLearnerCount++;
		}
		
		returnMap.put(Constant.ROOT_NODES, rootNodes);
		returnMap.put(Constant.ALPHAS, alphas);
		return returnMap;

	}
	
	public static Double validateTrainData(List<Node> rootNodes, List<Double> alphas,
			List<Integer> testDPList, Matrix dataMatrix){
		
		Integer numOfTestDP = testDPList.size();
		double dataArray[][] = dataMatrix.getArray();
		int count = rootNodes.size();
		int correctCount = 0;
		
		for(int i=0;i< numOfTestDP;i++){
		
			Double score = 0d;
			
			Integer testDP = testDPList.get(i);
			
			for(int j = 0; j < count; j++){	
				
				score += alphas.get(j) * 
						DecisionStamp.predictionValue(dataArray[testDP],rootNodes.get(j));
			
				double actualTargetValue = dataArray[testDP][Constant.SPAMBASE_DATA_TARGET_VALUE_INDEX];
				double predictedValue = 0d;
				
				if (score > 0)
					predictedValue = 1d;
				else
					predictedValue = -1d;
				
				if(actualTargetValue == predictedValue)
					correctCount++;
			}
			
		}
		
		System.out.println("Prediction Accuracy :" + ((double)correctCount/numOfTestDP));
		Double error = 1d - (double)correctCount/numOfTestDP;
		return error;
		
	}
	
	public void performAdaBoostOnUCIDS(){
		
		HashMap<Integer,HashMap<String,Double>> mapping = new HashMap<Integer,HashMap<String,Double>>();
		HashMap<String,Double> target = new HashMap<String,Double>();
		target.put("L", 1d);
		target.put("B", 2d);
		target.put("R", 3d);
		mapping.put(4, target);
		
		HashMap<Integer,String> attributeType = new HashMap<Integer,String>();
		attributeType.put(0, Constant.CATEGORICAL);
		attributeType.put(1, Constant.CATEGORICAL);
		attributeType.put(2, Constant.CATEGORICAL);
		attributeType.put(3, Constant.CATEGORICAL);
		
		
		HashMap<String,Matrix> matrixHashMap = fileOperations.fetchAttributesTargetFromFile(
				Constant.BOOSTING_FILE_PATH,Constant.MULTI_CLASS_FILE_NAME, 
				Constant.BAL_DATA_NUM_OF_DP, Constant.BAL_DATA_NUM_OF_FEATURES, 
				"\t",mapping);
		Matrix attributeMatrix = matrixHashMap.get(Constant.ATTRIBUTES);
		Matrix actualTargetMatrix = matrixHashMap.get(Constant.TARGET);
		
		System.out.println("Number of datapoints "+ attributeMatrix.getRowDimension());
		
		DecisionStamp decisionStamp = new DecisionStamp();
		HashMap<Integer,List<Integer>> dataPointPerFold = decisionStamp.
				formDifferentDataPerFold(Constant.BAL_DATA_NUM_OF_DP,Constant.NBR_OF_FOLDS);
		
		// Select Train and test data randomly
		// Train 80%
		// Test 20%
		
		int currentFoldCount = 0;
		List<Integer> trainDPList = null;
		List<Integer> testDPList = null;
		
		while(currentFoldCount < Constant.NBR_OF_FOLDS){
		
			// Apply "multiclass" solution : one vs the others separately for each class.
			HashMap<String,List<Integer>> hashMap  = generateTrainTestData(dataPointPerFold,currentFoldCount);
			trainDPList = hashMap.get(Constant.TRAIN_DP);
			testDPList = hashMap.get(Constant.TEST_DP);
			
			int count = 1;
			int nbrOfallowedWeakLeaner = 100;
			HashMap<Integer,HashMap<String,Object>> classifierPerClass = new HashMap<Integer,HashMap<String,Object>>(); 
			while(count <= Constant.BAL_NBR_OF_CLASS){
				
				Matrix updatedTargetMatrix = modifiyTarget((Matrix)actualTargetMatrix.clone(), count);
				HashMap<String,Object> returnedMap = performAdaboosting(attributeMatrix, updatedTargetMatrix, 
						trainDPList, attributeType, nbrOfallowedWeakLeaner,0,Constant.BAL_DATA_TARGET_VALUE_INDEX);
				classifierPerClass.put(count, returnedMap);
				count++;
				
			}
			
			// Validate Model on Test Data
			validateTestData(classifierPerClass, testDPList, 
					mergetAttributeTargetMatrix(attributeMatrix, actualTargetMatrix),
					Constant.BAL_DATA_TARGET_VALUE_INDEX);
			currentFoldCount++;
		}
		
	}
	
	public HashMap<String,List<Integer>> generateTrainTestData(
			HashMap<Integer,List<Integer>> hashMap, int currentFoldCount){
		
		HashMap<String,List<Integer>> dpHashMap = new HashMap<String,List<Integer>>();
		List<Integer> trainDPList = new ArrayList<Integer>();
		List<Integer> testDPList = new ArrayList<Integer>();
		
		Iterator<Map.Entry<Integer,List<Integer>>> iterator = hashMap.entrySet().iterator();
		
		while(iterator.hasNext()){
			
			Entry<Integer,List<Integer>> entry = iterator.next();
			if(currentFoldCount != entry.getKey()){
				trainDPList.addAll(entry.getValue());
			}else{
				testDPList.addAll(entry.getValue());
			}
		}
		dpHashMap.put(Constant.TRAIN_DP, trainDPList);
		dpHashMap.put(Constant.TEST_DP, testDPList);
		
		return dpHashMap;
	}
	
	public Matrix modifiyTarget(Matrix targetMatrix, Integer selectedClass){
		
		double targetValue[][] = targetMatrix.getArray();
		int nbrOfDP = targetMatrix.getRowDimension();
		
		for(int i = 0 ; i < nbrOfDP; i++){
			
			if(targetValue[i][0] != selectedClass){
				targetValue[i][0] = -1d;
			}
			else{
				targetValue[i][0] = 1d;
			}
		}
		
		return new Matrix (targetValue);
	}
	
	public static HashMap<Integer,Double> initializeClassScore(int nbrOfClass){
		
		HashMap<Integer,Double> scorePerClass = new HashMap<Integer,Double>();
		
		for(int i=1; i<=nbrOfClass; i++){
			scorePerClass.put(i, 0d);
		}
		
		return scorePerClass;
		
	}
	
	public static void validateTestData(HashMap<Integer,HashMap<String,Object>> classifierPerClass,
			List<Integer> testDPList, Matrix dataMatrix, Integer targetValueIndex){
		
		Integer numOfTestDP = testDPList.size();
		double dataArray[][] = dataMatrix.getArray();
		
		int correctCount = 0;
		
		for(int i=0;i< numOfTestDP;i++){
			
			Integer testDP = testDPList.get(i);
			int classCount = 1;
			
			HashMap<Integer,Double> scorePerClass = initializeClassScore(Constant.BAL_NBR_OF_CLASS);
			
			while(classCount <= Constant.BAL_NBR_OF_CLASS){
				
				HashMap<String,Object> returneMap = classifierPerClass.get(classCount);
				
				ArrayList<Node> rootNodes = (ArrayList<Node>)returneMap.get(Constant.ROOT_NODES);
				ArrayList<Double> alphas = (ArrayList<Double>)returneMap.get(Constant.ALPHAS);
				Double score = 0d;
				int count = rootNodes.size();
				
				for(int j = 0; j < count; j++){
					score += alphas.get(j) * 
							DecisionStamp.predictionValue(dataArray[testDP],rootNodes.get(j));
				}
				
				if(score > 0){
					scorePerClass.put(classCount, scorePerClass.get(classCount) + 1);
				}else{
					
					for(int n = 1; n <= Constant.BAL_NBR_OF_CLASS; n++){
						if(n != classCount)
							scorePerClass.put(n, scorePerClass.get(n) + 1);
					}
					
				}
				classCount++;
			}
			
			double actualTargetValue = dataArray[testDP][targetValueIndex];
			double predictedValue = 0d;
			
			// Find the class which has highest score
			
			predictedValue = (double)scorePerClass.entrySet().stream().max(
					(entry1, entry2) -> entry1.getValue() > entry2.getValue() ? 1 : -1).get().getKey();
			
			if(actualTargetValue == predictedValue)
				correctCount++;
		}
		
		System.out.println("Prediction Accuracy :" + ((double)correctCount/numOfTestDP));
		
	}
	
	
	public static void main(String args[]){
		
		AdaBoost adaBoost = new AdaBoost();
		adaBoost.performAdaBoostOnUCIDS();
		
	}

}


