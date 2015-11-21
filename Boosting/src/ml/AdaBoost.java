package ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import Jama.Matrix;



public class AdaBoost {
	
	static HashMap<Integer,Double> trainDataDistribution;
	FileOperations fileOperations;
	
	public AdaBoost(){
		fileOperations = new FileOperations();
	}
	
	
	public void performAdaBoosting(){
		
		Matrix dataMatrix = fileOperations.fetchDataPointsFromFile(
				Constant.SPAMBASE_DATA_FILE_PATH,Constant.SPAMBASE_DATA_FILE_NAME,
				Constant.NBR_OF_DP,Constant.NBR_OF_FEATURES + 1
				,Constant.COMMA_REGEX);
		
		DecisionStamp decisionStamp = new DecisionStamp(dataMatrix);
		HashMap<Integer,List<Integer>> dataPointPerFold = decisionStamp.formDifferentDataPerFold();
		
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
			
			ArrayList<Feature> features = decisionStamp.fetchFeaturePosThreshold(trainDPList);
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

				decisionStamp.formTrainDecisionTree(rootNode, features, trainDataDistribution);
				
				//Calculate the weighted error.
				// Evaluate Decision Tree
				HashMap<String, Object> returnMap = decisionStamp.calculateWeightedError(trainDPList, 
						rootNode, trainDataDistribution);
				
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
			validateTestData(rootNodes, alphas, testDPList, dataMatrix);
			System.out.println("Train Error Data");
			System.out.println(trainErrors);
			System.out.println("Round Error per Iteration");
			System.out.println(weightedErrors);
			break;
		}
		

		

	}
	
	public static void validateTestData(ArrayList<Node> rootNodes, ArrayList<Double> alphas,
			List<Integer> testDPList, Matrix dataMatrix){
		
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
				
				double actualTargetValue = dataArray[testDP][Constant.SPAMBASE_DATA_TARGET_VALUE_INDEX];
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
	
	public HashMap<String,Object> performAdaboosting(Matrix dataMatrix, List<Integer> trainDPList){
		
		int weakLearnerCount = 0;
		
		HashMap<String,Object> returnMap = new HashMap<String,Object>();
		
		DecisionStamp decisionStamp = new DecisionStamp(dataMatrix);
		ArrayList<Feature> features = decisionStamp.fetchFeaturePosThreshold(trainDPList);
		List<Node> rootNodes = new ArrayList<Node>();
		ArrayList<Double> alphas = new ArrayList<Double>();
		
		formInitialDataDistribution(trainDPList);
		
		while(weakLearnerCount < 20){
			
			//System.out.println("Weak Learner Count :" + weakLearnerCount);
			
			// Apply the weak learner on the training Data set.
			Node rootNode = new Node();
			rootNode.setDataPoints(trainDPList);
			
			decisionStamp.formTrainDecisionTree(rootNode, features, trainDataDistribution);

			//Calculate the weighted error.
			// Evaluate Decision Tree
			HashMap<String, Object> map = decisionStamp.calculateWeightedError(trainDPList, 
					rootNode, trainDataDistribution);
			
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
	
	
	public static void main(String args[]){
		
		AdaBoost adaBoost = new AdaBoost();
		adaBoost.performAdaBoosting();
		
	}

}


