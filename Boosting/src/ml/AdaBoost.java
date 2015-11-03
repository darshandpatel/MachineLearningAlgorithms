package ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import Jama.Matrix;



public class AdaBoost {
	
	static HashMap<Integer, Double> trainDataDistribution;
	
	public void performAdaBoosting(){
		
		
		
		DecisionTree decisionTree = new DecisionTree();
		decisionTree.formDifferentDataPerFold();
		
		// Number of fold will be created on the given Data Matrix;
		int numOfFolds = 10;
		int currentFoldCount = 0;
		
		// Total average accuracy of all the Decision Trees.
		Double sumOfAccuracy = 0d;
		
		while(currentFoldCount < numOfFolds){
			
			System.out.println("===============================");
			System.out.println("Fold count : " + currentFoldCount);
			
			HashMap<String,Matrix> matrixHashMap = decisionTree.formDataMatrixByFold(numOfFolds,currentFoldCount);
			Matrix trainDataMatrix = matrixHashMap.get(Constant.TRAIN);
			formInitialDataDistribution(trainDataMatrix.getRowDimension());
			ArrayList<Node> rootNodes = new ArrayList<Node>();
			ArrayList<Double> alphas = new ArrayList<Double>();
			int weakLearnerCount = 0;
			while(weakLearnerCount <= 100){
				
				System.out.println("Weak Learner Count :" + weakLearnerCount);
				
				// Apply the weak learner on the training Data set.
				Node rootNode = new Node();
				rootNode.setDataPoints(trainDataMatrix.getRowDimension());
				HashMap<String,Object> entropyMap = decisionTree.calculateEntropy(rootNode.getDataPoints(),
						trainDataDistribution);
				rootNode.setEntropy((Double)entropyMap.get(Constant.ENTROPY));
				
				rootNode.setLabelValue(
						((Integer)entropyMap.get(Constant.SPAM_COUNT) > 
						(Integer)entropyMap.get(Constant.HAM_COUNT))?1d:0d);
				
				decisionTree.formTrainDecisionTree(rootNode,trainDataMatrix, trainDataDistribution);
				
				//System.out.println("Print the Decision Tree");
				//decisionTree.printDecisionTree(rootNode);

				//Calculate the weighted error.
				// Evaluate Decision Tree
				HashMap<String, Object> returnMap = decisionTree.calculateWeightedError(trainDataMatrix, rootNode, 
						trainDataDistribution);
				
				Double weighedError = (Double)returnMap.get(Constant.ERROR_VALUE);
				
				// Calculate the alpha
				
				double alpha = (double)(1d/2d) * (Math.log((1 - weighedError) / weighedError));
				returnMap.put(Constant.ALPHA_VALUE, alpha);

				// Update the data distribution
				updateDataDistribution(returnMap);	
				rootNodes.add(rootNode);
				alphas.add(alpha);
				
				weakLearnerCount++;
			}
			/*
			Matrix testDataMatrix = matrixHashMap.get(Constant.TEST);
			
			// Accumulate Results
			Double accuracy = decisionTree.evaluateTestDataSet(testDataMatrix, rootNode);
			sumOfAccuracy += accuracy;
			
			// Root Node for the current Decision Tree
			currentFoldCount++;	
			break;
			*/
		}
		
		// Print the Results
		System.out.println("Final accuracy : "+ sumOfAccuracy/numOfFolds);

	}
	
	public static void formInitialDataDistribution(int nbrOfTrainDP){
		
		trainDataDistribution = new HashMap<Integer, Double>();
		
		for(int i = 0; i < nbrOfTrainDP; i++){
			trainDataDistribution.put(i, (double)(1.0d/nbrOfTrainDP));
		}
		
	}
	
	public static void updateDataDistribution(HashMap<String, Object> returnMap){
		
		Double alpha = (Double) returnMap.get(Constant.ALPHA_VALUE);
		List<Integer> misClassifiedDP = (List<Integer>) returnMap.get(Constant.MISCLASSIFIED_DP);
		List<Integer> correctlyClassifiedDP = (List<Integer>) returnMap.get(Constant.CORRECTLY_CLASSIFIED_DP);
		
		int noOfTrainDP = misClassifiedDP.size() + correctlyClassifiedDP.size();
		Double updatedDataDistirbution;
		Double sumOfDistributionValue = 0d;
		
		for(Integer dataPoint : misClassifiedDP){
			
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
		for(int i = 0; i < noOfTrainDP; i++){
			trainDataDistribution.put(i, trainDataDistribution.get(i)/sumOfDistributionValue);
		}
	}
	
	
	public static void main(String args[]){
		
		AdaBoost adaBoost = new AdaBoost();
		adaBoost.performAdaBoosting();
		
	}

}


