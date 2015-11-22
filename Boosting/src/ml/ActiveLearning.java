package ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.IntSupplier;
import java.util.stream.IntStream;

import com.google.common.primitives.Ints;

import Jama.Matrix;

public class ActiveLearning {
	
	int[] shareOfTrainingData = {5,10,15,20,30,50};
	
	FileOperations fileOperations;
	Matrix dataMatrix;
	Integer nbrOfDP;
	AdaBoost adaBoost;
	
	public ActiveLearning(){
		fileOperations = new FileOperations();
		adaBoost = new AdaBoost();
	}
	
	@SuppressWarnings("unchecked")
	public void performActiveLearning(){
		
		int perOfNewTrainDP = 2;
		
		dataMatrix = fileOperations.fetchDataPointsFromFile(
				Constant.SPAMBASE_DATA_FILE_PATH,Constant.SPAMBASE_DATA_FILE_NAME,
				Constant.NBR_OF_DP,Constant.NBR_OF_FEATURES + 1
				,Constant.COMMA_REGEX);
		nbrOfDP = dataMatrix.getRowDimension();
		
		int count = shareOfTrainingData.length;
		
		for(int i = 0 ; i < count; i++){
			
			int percentOfTrainingData = shareOfTrainingData[i];
			
			HashMap<String,List<Integer>> hashMap = retrieveTrainTestDP(percentOfTrainingData);
			
			List<Integer> trainDPList = hashMap.get(Constant.TRAIN_DP);
			List<Integer> testDPList = hashMap.get(Constant.TEST_DP);
			int nbrOfAllowedWeakLearner = 20;
			do{
				
				int nbrOfTestDP = testDPList.size();
				int nbrOfTrainDP = trainDPList.size();
				
				System.out.println("Number of training data "+ nbrOfTrainDP);
				
				HashMap<String, Object> resultMap = adaBoost.performAdaboosting(dataMatrix, trainDPList, nbrOfAllowedWeakLearner);
				
				List<Node> rootNodes = (List<Node>) resultMap.get(Constant.ROOT_NODES);
				List<Double> alphas = (List<Double>) resultMap.get(Constant.ALPHAS);
				
				validateTestData(rootNodes, alphas, testDPList, dataMatrix);
				
				Map<Integer, Double> sortedTestDPScore = analyzeTestData(testDPList, 
						rootNodes, alphas);
				
				
				int nbrOfNewTrainDP = nbrOfTestDP * perOfNewTrainDP / 100;
				System.out.println("Newly added train data # :"+ nbrOfNewTrainDP);
				
				if(nbrOfTrainDP + nbrOfNewTrainDP < (nbrOfDP * 0.5)){
					
					hashMap = updateTrainTestDP(trainDPList, sortedTestDPScore, nbrOfNewTrainDP);
					trainDPList = hashMap.get(Constant.TRAIN_DP);
					testDPList = hashMap.get(Constant.TEST_DP);
					
				}else
					break;
				
			}while(true);
			break;
			
		}
		
	}
	
	public HashMap<String,List<Integer>> retrieveTrainTestDP(int percentOfTrainingData){
		
		HashMap<String,List<Integer>> hashMap = new HashMap<String,List<Integer>>();
		
		int nbrOfAllowedTrainDP = nbrOfDP * percentOfTrainingData / 100;
		int randomDP[] = new Random().ints(0,nbrOfDP).distinct()
                .limit(nbrOfDP).toArray();
		
		int trainDP[] = Arrays.copyOfRange(randomDP,0,nbrOfAllowedTrainDP);
		int testDP[] = Arrays.copyOfRange(randomDP,nbrOfAllowedTrainDP,nbrOfDP);
		
		hashMap.put(Constant.TRAIN_DP, new ArrayList(Ints.asList(trainDP)));
		hashMap.put(Constant.TEST_DP, new ArrayList(Ints.asList(testDP)));
		
		return hashMap;
		
	}
	
	public Map<Integer, Double> analyzeTestData(List<Integer> testDPList, List<Node> rootNodes, 
			List<Double> alphas){
		
		Map<Integer, Double> testDPScore = new HashMap<Integer, Double>();
		
		double dataArray[][] = dataMatrix.getArray();
		
		int count = rootNodes.size();
		Integer numOfTestDP = testDPList.size();
		
		for(int i = 0;i< numOfTestDP;i++){
			
			Double score = 0d;
			Integer testDP = testDPList.get(i);
			
			for(int j = 0; j < count; j++){
				score += alphas.get(j) * 
						DecisionStamp.predictionValue(dataArray[testDP],rootNodes.get(j));
			}
			
			testDPScore.put(testDP, Math.abs(score));
				
		}
		
		Map<Integer, Double> sortedTestDPScore = sortByComparator(testDPScore);
		return sortedTestDPScore;
		
	}
	
	private static Map<Integer, Double> sortByComparator(Map<Integer, Double> unsortMap) {

		// Convert Map to List
		List<Map.Entry<Integer, Double>> list = 
			new LinkedList<Map.Entry<Integer, Double>>(unsortMap.entrySet());

		// Sort list with comparator, to compare the Map values
		Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>() {
			public int compare(Map.Entry<Integer, Double> o1,
                                           Map.Entry<Integer, Double> o2) {
				return (o1.getValue()).compareTo(o2.getValue());
			}
		});

		// Convert sorted map back to a Map
		Map<Integer, Double> sortedMap = new LinkedHashMap<Integer, Double>();
		for (Iterator<Map.Entry<Integer, Double>> it = list.iterator(); it.hasNext();) {
			Map.Entry<Integer, Double> entry = it.next();
			sortedMap.put(entry.getKey(), entry.getValue());
		}
		return sortedMap;
	}
	
	public HashMap<String,List<Integer>> updateTrainTestDP(List<Integer> trainDPList,
			Map<Integer, Double> sortedTestDPScore,
			int nbrOfNewTrainDP){
		
		HashMap<String,List<Integer>> map = new HashMap<String,List<Integer>>();
		List<Integer> newTestDPList = new ArrayList<Integer>();
		
		Iterator<Map.Entry<Integer, Double>> iterator = sortedTestDPScore.entrySet().iterator();
		
		int count = 1;
		while(iterator.hasNext()){
			
			Integer dataPoint = iterator.next().getKey();
			// Add data point to training data point list
			if(count <= nbrOfNewTrainDP){
				trainDPList.add(dataPoint);
			}else{
				newTestDPList.add(dataPoint);
			}
			
			count++;
		}
		
		map.put(Constant.TEST_DP, newTestDPList);
		map.put(Constant.TRAIN_DP, trainDPList);
		
		return map;
	}
	
	public static void validateTestData(List<Node> rootNodes, List<Double> alphas,
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
			
		
		System.out.println("Prediction Accuracy :" + ((double)correctCount/numOfTestDP));
		
	}
	
	
	public static void main(String args[]){
		
		ActiveLearning activeLearning = new ActiveLearning();
		activeLearning.performActiveLearning();
		
	}

}
