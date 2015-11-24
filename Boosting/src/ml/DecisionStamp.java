package ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import Jama.Matrix;

import com.google.common.primitives.Doubles;

/**
 * 
 * @author Darshan
 *
 */
public class DecisionStamp{
	
	Matrix dataMatrix;
	HashMap<Integer,List<Integer>> dataPointPerFold;
	
	public DecisionStamp(){
		
	}
	
	public DecisionStamp(Matrix dataMatrix){
		this.dataMatrix = dataMatrix;
	}
	
	
	public HashMap<Integer,List<Integer>> formDifferentDataPerFold(int nbrOfDP, int nbrOfFolds){
		
		dataPointPerFold = new HashMap<Integer,List<Integer>>();
		
		int numOfDPPerFold = nbrOfDP / nbrOfFolds;
		int extraDP = nbrOfDP % nbrOfFolds;
		
		for(int i = 0; i < nbrOfFolds; i++){
			List<Integer> numbers = IntStream.iterate(i, n -> n + 10).limit(numOfDPPerFold)
					.boxed().collect(Collectors.toList());
			dataPointPerFold.put(i, numbers);
		}
		
		int remainingDP = nbrOfDP - extraDP;
		dataPointPerFold.get(nbrOfFolds-1).addAll(
				IntStream.iterate(remainingDP, n -> n + 1).limit(extraDP).boxed().collect(Collectors.toList()));
		dataPointPerFold.get(nbrOfFolds-1);
		return dataPointPerFold;
		
	}
	
	
	public HashMap<String,List<Integer>> formTrainTestDPByFold(Integer numOfFolds,Integer currentFoldCount){
		
		HashMap<String,List<Integer>> matrixHashMap = new HashMap<String,List<Integer>>();
		
		List<Integer> testDPList = new ArrayList<Integer>();
		List<Integer> trainDPList = new ArrayList<Integer>();
		
		for(int i = 0; i < Constant.NBR_OF_FOLDS; i++){
			
			if(i != currentFoldCount){
				
				List<Integer> dataPoints = dataPointPerFold.get(i);
				Iterator<Integer> dataPointIterator = dataPoints.iterator();
				
				while(dataPointIterator.hasNext()){
					
					int rowCount = dataPointIterator.next();
					trainDPList.add(rowCount);
				}
				
			}else{
				
				List<Integer> dataPoints = dataPointPerFold.get(i);
				Iterator<Integer> dataPointIterator = dataPoints.iterator();
				
				while(dataPointIterator.hasNext()){
					
					int rowCount = dataPointIterator.next();
					testDPList.add(rowCount);
				}
				
				
			}
			
		}
		
		matrixHashMap.put(Constant.TRAIN_DP, trainDPList);
		matrixHashMap.put(Constant.TEST_DP, testDPList);
		
		return matrixHashMap;
	}
	
	public HashMap<String,Object> calculateEntropy(List<Integer> dataPoints, 
			HashMap<Integer,Double> dataDistribution){
		
		double sumOfSpamDataDistribution = 0;
		double sumOfNonSpamDataDistribution = 0;
		
		HashMap<String,Object> entropyMap = new HashMap<String,Object>();
		//System.out.println("Matrix # of rows :"+dataMatrix.getRowDimension());
		//System.out.println("Matrix # of columns :"+dataMatrix.getColumnDimension());
		
		for(Integer dataPoint : dataPoints){
			if(dataMatrix.get(dataPoint,Constant.SPAMBASE_DATA_TARGET_VALUE_INDEX) == 1){
				sumOfSpamDataDistribution += dataDistribution.get(dataPoint);
			}
			else{
				sumOfNonSpamDataDistribution += dataDistribution.get(dataPoint);
			}
		}

		Double entropy = -((sumOfSpamDataDistribution * log2(sumOfSpamDataDistribution)) + 
				(sumOfNonSpamDataDistribution * log2(sumOfNonSpamDataDistribution)));
		
		//System.out.println("Entropy : "+entropy);
		
		entropyMap.put(Constant.ENTROPY, entropy);
		entropyMap.put(Constant.SPAM_SUM_OF_DIST, sumOfSpamDataDistribution);
		entropyMap.put(Constant.NON_SPAM_SUM_OF_DIST, sumOfNonSpamDataDistribution);
		
		return entropyMap;
	}
	
	/**
	 * 
	 * @param n : Number
	 * @return the log (base 2) of the given number
	 */
	private static double log2(double n)
	{
		if(n != 0)
			return (Math.log(n) / Math.log(2));
		else
			return 0d;
	}
	
	
	public void formTrainDecisionTree(Node rootNode, ArrayList<Feature> features,
			HashMap<Integer,Double> dataDistribution, int treeDepthLimit, int targetIndexValue){
		
		Integer exploredNodeCount = 0;
		Integer exploredNodeLimit = 1;
		for(int i = 0; i < treeDepthLimit; i++){
			exploredNodeLimit += (int)Math.pow(2,i);
		}
		
		Queue<Node> nodeQueue = new LinkedList<Node>();
		nodeQueue.add(rootNode);
		
		// Form the tree until Node queue is not empty and the # of explored node 
		// is less the # of explored node limit
		
		while( (nodeQueue.size() > 0) && (exploredNodeCount < exploredNodeLimit)){
			
			Node currentNode = nodeQueue.poll();
			exploredNodeCount++;
		
			splitNodeByBestFeaturethreshold(currentNode, dataDistribution, features, targetIndexValue);
			
			if(currentNode.getLeftChildNode() != null){
				
				Node leftNode = currentNode.getLeftChildNode();
				//if(leftNode.getEntropy() > Constant.HOUSING_DATA_ERROR_THRESHOLD)
				nodeQueue.add(leftNode);
				
			}
			if(currentNode.getRightChildNode() != null){
				
				Node rightNode = currentNode.getRightChildNode();
				//if(rightNode.getEntropy() > Constant.HOUSING_DATA_ERROR_THRESHOLD)
				nodeQueue.add(rightNode);
				
			}
		}
		
	}
	
	public void splitNodeByBestFeaturethreshold(Node node, HashMap<Integer, Double> dataDistribution,
			ArrayList<Feature> features, Integer targetIndexValue){
		
		List<Integer> currentNodeDataPoints = node.getDataPoints();
		Integer noOfCurrentNodeDataPoints = currentNodeDataPoints.size();
		
		HashMap<Integer,Double> infoGainPerFeatureBestThreshold = new HashMap<Integer,Double>();
		HashMap<Integer,Integer> bestThresholdIndexPerFeature = new HashMap<Integer,Integer>();
		
		HashMap<Integer,ArrayList<Integer>> leftSideDPForFeaturesBestThreshold = 
				new HashMap<Integer,ArrayList<Integer>>();
		HashMap<Integer,ArrayList<Integer>> rightSideDPForFeatureBestThreshold = 
				new HashMap<Integer,ArrayList<Integer>>();
		
		for(Feature feature : features){
			
			//System.out.println("*********************FEATURE STARTS********************");
			
			HashMap<Integer,ArrayList<Integer>> leftSideDPPerThreshold = 
					new HashMap<Integer,ArrayList<Integer>>();
			HashMap<Integer,ArrayList<Integer>> rightSideDPPerThreshold = 
					new HashMap<Integer,ArrayList<Integer>>();
			
			int featureIndex = feature.getIndex();

			List<Double> thresholdValues = feature.getThresholdValues();
			//System.out.println("Feature index :"+feature.getIndex());
			//System.out.println("Feature Threshold :" + thresholdValues.toString());
			int noOfThresholdValues = thresholdValues.size();
			
			if(feature.getType().equals(Constant.NUMERIC)){
				
				for(int i = 0 ; i < noOfThresholdValues;i++){
					
					Double thresholdValue = thresholdValues.get(i);
					for(int k = 0 ; k < noOfCurrentNodeDataPoints ; k++){
						
						Integer dataPoint = currentNodeDataPoints.get(k);
						Double trainFeatureValue = dataMatrix.get(dataPoint,featureIndex);
						
						if(trainFeatureValue < thresholdValue){
							
							if(leftSideDPPerThreshold.containsKey(i)){
								//System.out.println("Left side node Threshold value Index: "+i+" 
								//, size is : "+leftSideDPPerThreshold.get(i).size());
								leftSideDPPerThreshold.get(i).add(dataPoint);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(dataPoint);
								leftSideDPPerThreshold.put(i,dataPoints);
							}
						}else{
							if(rightSideDPPerThreshold.containsKey(i)){
								//System.out.println("Right side node Threshold value index : "+i+"
								// Threshold value "+trainFeatureValue+", size is : " + 
								//rightSideDPPerThreshold.get(i).size());
								rightSideDPPerThreshold.get(i).add(dataPoint);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(dataPoint);
								rightSideDPPerThreshold.put(i,dataPoints);
							}
						}
					}
				}
			}else if(feature.getType().equals(Constant.BINARY_NUM)){
				
				for(int i= 0 ; i< noOfThresholdValues;i++){
					
					for(int k = 0 ; k < noOfCurrentNodeDataPoints ; k++){
					
						Integer dataPoint = currentNodeDataPoints.get(k);
						Double trainFeatureValue = dataMatrix.get(dataPoint,featureIndex);
					
						//System.out.println("In side binary");
						//System.out.println("# of different value "+ noOfThresholdValues);
					
						if(trainFeatureValue == 0){
							if(leftSideDPPerThreshold.containsKey(i)){
								leftSideDPPerThreshold.get(i).add(dataPoint);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(dataPoint);
								leftSideDPPerThreshold.put(i,dataPoints);
							}
						}else{
							if(rightSideDPPerThreshold.containsKey(i)){
								rightSideDPPerThreshold.get(i).add(dataPoint);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(dataPoint);
								rightSideDPPerThreshold.put(i,dataPoints);
							}
						}
					}
				}
				
			}else if(feature.getType().equals(Constant.CATEGORICAL)){
				
				for(int i= 0 ; i< noOfThresholdValues;i++){
					
					Double thresholdValue = thresholdValues.get(i);
					for(int k = 0 ; k < noOfCurrentNodeDataPoints ; k++){
					
						Integer dataPoint = currentNodeDataPoints.get(k);
						Double trainFeatureValue = dataMatrix.get(dataPoint,featureIndex);
					
						//System.out.println("In side binary");
						//System.out.println("# of different value "+ noOfThresholdValues);
					
						if(trainFeatureValue == thresholdValue){
							if(leftSideDPPerThreshold.containsKey(i)){
								leftSideDPPerThreshold.get(i).add(dataPoint);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(dataPoint);
								leftSideDPPerThreshold.put(i,dataPoints);
							}
						}else{
							if(rightSideDPPerThreshold.containsKey(i)){
								rightSideDPPerThreshold.get(i).add(dataPoint);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(dataPoint);
								rightSideDPPerThreshold.put(i,dataPoints);
							}
						}
					}
				}
				
			}
			
			//TODO Only single method can return the best feature index
			
			// Calculate Information Gain
			
			HashMap<Integer,Double> informationGainPerThresholdValue = calculateInformationGain(feature, 
					leftSideDPPerThreshold, rightSideDPPerThreshold, dataDistribution, targetIndexValue);
			
			Iterator<Entry<Integer, Double>> igIterator = 
					informationGainPerThresholdValue.entrySet().iterator();
			Double highestIG = Double.NEGATIVE_INFINITY;
			Integer bestThresholdIndex=0;
			
			while(igIterator.hasNext()){
				Entry<Integer, Double> entry = igIterator.next();
				if(entry.getValue() > highestIG){
					highestIG = entry.getValue();
					bestThresholdIndex = entry.getKey();
				}
			}
			
			bestThresholdIndexPerFeature.put(featureIndex,bestThresholdIndex);
			infoGainPerFeatureBestThreshold.put(featureIndex,highestIG);
			
			/*
			System.out.println("# of threshold values 	: " + thresholdValues.size());
			System.out.println("Information Gain		: " + infoGain);
			System.out.println("Feature index 			: " + featureIndex);
			System.out.println("Feature type 			: " + feature.getType());
			System.out.println("Threshold value index 	: " + bestThresholdIndex);
			System.out.println("Threshold value 		: " + 
			thresholdValues.get(bestThresholdIndex));
			*/
			
			if(leftSideDPPerThreshold.get(bestThresholdIndex) != null){
				//System.out.println("# of datapoint on left side: " + 
				//	leftSideDPPerThreshold.get(bestThresholdIndex).size());
				leftSideDPForFeaturesBestThreshold.put(featureIndex,
						leftSideDPPerThreshold.get(bestThresholdIndex));
			}
			if(rightSideDPPerThreshold.get(bestThresholdIndex) != null){
				//System.out.println("# of datapoint on right side: " + 
				//	rightSideDPPerThreshold.get(bestThresholdIndex).size());
				rightSideDPForFeatureBestThreshold.put(featureIndex,
						rightSideDPPerThreshold.get(bestThresholdIndex));
			}
			
		}
		
		
		createChildNodes(node,features,infoGainPerFeatureBestThreshold, bestThresholdIndexPerFeature, 
				leftSideDPForFeaturesBestThreshold,rightSideDPForFeatureBestThreshold, dataDistribution);
		
	}
	
	
	public HashMap<Integer,Double> calculateInformationGain(Feature feature,
			HashMap<Integer,ArrayList<Integer>> leftDataPoints,
			HashMap<Integer,ArrayList<Integer>> rightDataPoints,
			HashMap<Integer, Double> dataDistribution, Integer targetIndexValue){
		
		HashMap<Integer,Double> informationGainPerThresold = new HashMap<Integer,Double>();
		
		// Left side data point will be labeled as -1
		// Right side data point will be labeled as 1
		double leftSideLabel = -1d;
		double rightSideLabel = 1d;
		
		List<Double> thresholdValues = feature.getThresholdValues();
		
		int noOfThresholdValues = thresholdValues.size();
		
		for(int i =0 ; i < noOfThresholdValues; i++){
			
			Double error = 0d;
			if(leftDataPoints.containsKey(i)){
				error += calculateError(leftDataPoints.get(i), dataDistribution, leftSideLabel, targetIndexValue);
			}
			
			if(rightDataPoints.containsKey(i)){
				error += calculateError(rightDataPoints.get(i), dataDistribution, rightSideLabel, targetIndexValue);
			}
			
			// In the binary classifier 0.9 error is actual 0.1 error if you flip the classifier label
			// Information gain is |0.5  - error|
			informationGainPerThresold.put(i, Math.abs(0.5 - error));
			
		}
		
		return informationGainPerThresold;
		
	}
	
	public Double calculateError(ArrayList<Integer> dataPoints, HashMap<Integer, Double> dataDistribution,
			Double labelValue, Integer targetIndexValue){
		
		Double error = 0d;
		
		for(Integer dataPoint : dataPoints){
			
			if(labelValue != dataMatrix.get(dataPoint, targetIndexValue)){
				error += dataDistribution.get(dataPoint);
			}
			
		}
		return error;
	}
	
	public HashMap<Integer,Double> calculateEntropy(Feature feature,
			HashMap<Integer,ArrayList<Integer>> leftDataPoints,
			HashMap<Integer,ArrayList<Integer>> rightDataPoints,
			HashMap<Integer, Double> dataDistribution){
			
		HashMap<Integer,Double> entropyPerThreshold = new HashMap<Integer,Double>();
		List<Double> thresholdValues = feature.getThresholdValues();
		
		int noOfThresholdValues = thresholdValues.size();
		
		for(int i =0 ; i < noOfThresholdValues; i++){
			
			Double totalChildNodeEntroy = 0d;
			Double entropyOfLeftChild = 0d;
			
			if(leftDataPoints.containsKey(i)){
				//System.out.println(leftDataPoints.get(i).toString());
				// Normalize the distribution based upon the subset of DataPoints
				
				HashMap<String,Object> entropyLeftSideMap = calculateEntropy(leftDataPoints.get(i),
								normalizeDataDistribution(leftDataPoints.get(i),dataDistribution));
				entropyOfLeftChild = (Double)entropyLeftSideMap.get(Constant.ENTROPY);
				totalChildNodeEntroy += (entropyOfLeftChild * sumDataDistribution(leftDataPoints.get(i),dataDistribution));
				//totalChildNodeEntroy += (entropyOfLeftChild);
			}
			//System.out.println("Left side entropy : "+entropyOfLeftChild);
			
			Double entropyRightChild = 0d;
			if(rightDataPoints.containsKey(i)){
				//System.out.println(rightDataPoints.get(i).toString());
				HashMap<String,Object> entropyRightSideMap = 
						calculateEntropy(rightDataPoints.get(i),
								normalizeDataDistribution(rightDataPoints.get(i),dataDistribution));
				entropyRightChild = (Double)entropyRightSideMap.get(Constant.ENTROPY);
				totalChildNodeEntroy += (entropyRightChild * sumDataDistribution(rightDataPoints.get(i),dataDistribution));
				//totalChildNodeEntroy += (entropyRightChild);
			}
			//System.out.println("Right side entropy : "+entropyRightChild);
			
			//System.out.println("Threshold Index : "+ i +"\t Threshold value : "+thresholdValues.get(i)
			//		+ "\t Error Value : "+error);
			entropyPerThreshold.put(i, totalChildNodeEntroy);
			
		}
		return entropyPerThreshold;
	}
	
	public HashMap<Integer, Double> normalizeDataDistribution(List<Integer> dataPoints,
			HashMap<Integer, Double> dataDistribution){
		
		HashMap<Integer, Double> updatedDataDistribution = new HashMap<Integer, Double>();
		
		Iterator<Integer> iterator = dataPoints.iterator();
		
		double sum = 0d;
		
		while(iterator.hasNext()){
			Integer dataPoint = iterator.next();
			double value = dataDistribution.get(dataPoint);
			sum += value;
		}
		
		iterator = dataPoints.iterator();
		while(iterator.hasNext()){
			Integer dataPoint = iterator.next();
			updatedDataDistribution.put(dataPoint,dataDistribution.get(dataPoint)/sum);
		}
		
		return updatedDataDistribution;
	}
	
	
	public ArrayList<Feature> fetchFeaturePosThreshold(List<Integer> dataPoints, 
			HashMap<Integer,String> attributeMapping){
		
		int noOfColumns = dataMatrix.getColumnDimension();
		
		HashMap<Integer,List<Double>> featurePosThreshold = 
				new HashMap<Integer,List<Double>>(Constant.NBR_OF_FEATURES);
			
		for(int i=0; i < (noOfColumns -1);i++){
			
			for(Integer row : dataPoints){
				
				//System.out.println("value of i is : " + i+ " : "+parts[i]);
				//String featureName = Constant.features.get(i);
				if(featurePosThreshold.containsKey(i)){
					Double value = dataMatrix.get(row,i);
					List<Double> values = featurePosThreshold.get(i);
					if(!values.contains(value))
						values.add(value);
					
				}else{
					
					List<Double> values = new ArrayList<Double>();
					values.add(dataMatrix.get(row,i));
					featurePosThreshold.put(i,values);
				}
			}
			
			if(attributeMapping.get(i) == Constant.NUMERIC){
				List<Double> values = featurePosThreshold.get(i);
				double minValue = values.stream().mapToDouble(p -> p).min().getAsDouble();
				values.add(minValue - 1);
				
				double maxValue = values.stream().mapToDouble(p -> p).max().getAsDouble();
				values.add(maxValue + 1);
			}
			
		}
		
		return createFeatures(featurePosThreshold,attributeMapping);
		
	}
	
	/**
	 * 
	 * @param featurePosThreshold : A HashMap which has key as feature index and
	 * value as the possible threshold value
	 * @return The ArrayList of feature objects.
	 */
	private ArrayList<Feature> createFeatures(HashMap<Integer,List<Double>> 
			featuresPosThreshold, HashMap<Integer,String> attributeMapping){
	
		ArrayList<Feature> features = new ArrayList<Feature>();
		int nbrOfFeature = attributeMapping.size();
		for(int i = 0; i < nbrOfFeature;i++){
			
			String featureCtg = attributeMapping.get(i);
			
			List<Double> calculatedFeaturePosCriValues = featuresPosThreshold.get(i);
			
			if(featureCtg.equals(Constant.NUMERIC)){
				Collections.sort(calculatedFeaturePosCriValues);
				calculatedFeaturePosCriValues = filterFeaturePosThreshold
						(calculatedFeaturePosCriValues);
			}
			
			Feature feature = new Feature(featureCtg,calculatedFeaturePosCriValues,i);
			//System.out.println("Feature values" + calculatedFeaturePosCriValues.toString());
			features.add(feature);
		}
		
		return features;
	}
	
	/**
	 * This method modifies the given unique filter threshold values(ArrayList).
	 * Basically it takes the average of two existing threshold value and makes that average value 
	 * a new threshold values
	 * @param calculatedFeaturePosCriValues : unique filter threshold values(ArrayList)
	 * @return the ArrayList of new threshold value for the given threshold value ArrayList
	 */
	public List<Double> filterFeaturePosThreshold 
				(List<Double> calculatedFeaturePosThresholdValues){
		
		
		int length = calculatedFeaturePosThresholdValues.size();
		if(length < 2){
			return calculatedFeaturePosThresholdValues;
		}else{
			ArrayList<Double> filterdPosThresholdValue= new ArrayList<Double>();
			for(int i=0 ; (i+1) < length; i++){
				Double filteredValue = ((calculatedFeaturePosThresholdValues.get(i) + 
						calculatedFeaturePosThresholdValues.get(i+1))/2);
				filterdPosThresholdValue.add(filteredValue);
			}
			calculatedFeaturePosThresholdValues.clear();
			return filterdPosThresholdValue;
		}
		
	}
	
	
	public void createChildNodes(Node node,ArrayList<Feature> features, 
			HashMap<Integer,Double> infoGainPerFeatureBestThreshold,
			HashMap<Integer,Integer> bestThresholdIndexPerFeature,
			HashMap<Integer,ArrayList<Integer>> leftSideDPForFeaturesBestThreshold,
			HashMap<Integer,ArrayList<Integer>> rightSideDPForFeatureBestThreshold,
			HashMap<Integer, Double> dataDistribution){
		
		Double higherstInfoGain = 0d;
		Integer bestFeatureIndex = 0;
		
		Iterator<Entry<Integer, Double>> infoGainIterator = 
				infoGainPerFeatureBestThreshold.entrySet().iterator();
		Double maxValue = Double.NEGATIVE_INFINITY;
		while(infoGainIterator.hasNext()){
			
			Entry<Integer, Double> entry = infoGainIterator.next();
			if(entry.getValue() > maxValue){
				maxValue = entry.getValue();
				bestFeatureIndex = entry.getKey();
			}
		}
		if(maxValue != Double.NEGATIVE_INFINITY)
			higherstInfoGain = maxValue;
		
		//System.out.println("Best Feature index is	: "+ bestFeatureIndex);
		//System.out.println("Information Gained is	: "+ higherstInfoGain);
		
		Feature bestFeature = features.get(bestFeatureIndex);
		
		//if(higherstInfoGain > Constant.SPAMBASE_DATA_INFO_GAIN_THRESHOLD){
			
		Integer bestThresholdValueIndex = bestThresholdIndexPerFeature.get(bestFeatureIndex);
		node.setThresholdValue(bestFeature.getThresholdValues().get(bestThresholdValueIndex));
		node.setFeatureIndex(bestFeatureIndex);
			//System.out.println("More than threshold information gain");
			
			//System.out.println("Best Threshold value	:" + bestFeature.getThresholdValues()
			//	.get(bestThresholdIndexPerFeature.get(bestFeatureIndex)));
			
		Node leftNode = new Node();
		leftNode.setLabelValue(-1d);
		node.setLeftChildNode(leftNode);
		
		Node rightNode = new Node();
		rightNode.setLabelValue(1d);
		node.setRightChildNode(rightNode);
			
		//}
		
	}
	
	/**
	 * This method print the important information of the given Decision Tree
	 * Node
	 * @param node : Decision Tree Root Node
	 */
	public void printDecisionTree(Node rootNode){
		
		Queue<Node> nodeQueue =  new LinkedList<Node>();
		nodeQueue.add(rootNode);
		Integer count = 1;
		while(nodeQueue.size() > 0){
			
			Node node = nodeQueue.poll();
			
			/*
			System.out.printf("%30s\t\t\t:\t%d\n","Node number",count);
			System.out.printf("%30s\t\t\t:\t%f\n","Node Entropy",node.getEntropy());
			System.out.printf("%30s\t\t\t:\t%d\n","Feature index",node.getFeatureIndex());
			System.out.printf("%30s\t\t\t:\t%f\n","Feature Threshold Value",node.getThresholdValue());
			System.out.printf("%30s\t\t\t:\t%f\n","Label Value",node.getLabelValue());
			System.out.printf("%30s\t\t\t:\t%f\n","Parent Node number",Math.floor(count/2));
			System.out.printf("%30s\t\t\t:\t%d\n\n",
			"Number of Data Points",node.getDataPoints().size());
			*/
			
			if(node.getLeftChildNode() != null){
				nodeQueue.add(node.getLeftChildNode());
			}
			if(node.getRightChildNode() != null){
				nodeQueue.add(node.getRightChildNode());
			}
			count++;
		}
	}
	
	
	public HashMap<String, Object> calculateWeightedError(List<Integer> trainDPList, 
			Node rootNode, HashMap<Integer,Double> dataDistribution, Integer targetValueIndex){
		
		HashMap<String, Object> returnObj = new HashMap<String, Object>();
		double dataArray[][] = dataMatrix.getArray();
		List<Integer> misClassifiedDataPoints = new ArrayList<Integer>();
		List<Integer> correctlyClassifiedDataPoints = new ArrayList<Integer>();
		
		Double error = 0d;
		Integer numOfDP = trainDPList.size();
		Integer dpIndex = 0;
		for(int i=0;i< numOfDP;i++){
			
			dpIndex = trainDPList.get(i);
			if(validatePredictionValue(dataArray[dpIndex],rootNode, targetValueIndex) == false){
				
				error += dataDistribution.get(dpIndex);
				misClassifiedDataPoints.add(dpIndex);
				
			}
			else{
				correctlyClassifiedDataPoints.add(dpIndex);
			}
		}
		
		returnObj.put(Constant.ERROR_VALUE, error);
		returnObj.put(Constant.MISCLASSIFIED_DP, misClassifiedDataPoints);
		returnObj.put(Constant.CORRECTLY_CLASSIFIED_DP, correctlyClassifiedDataPoints);
		
		return returnObj;
		
	}
	

	/**
	 * 
	 * @param dataValue : The single Data Point from the test Data Matrix.
	 * @param rootNode	: The root Node of the Decision Tree
	 * @param targetValueIndex : The index of target value in Data Matrix 
	 * @return			: 1 if the given decision tree can predict the target
	 * value of the given data point correctly otherwise return 0
	 */
	public static Boolean validatePredictionValue(double dataValue[],Node rootNode, Integer targetValueIndex){
		
		Node node = rootNode;
		
		while(true){

			if(node.getFeatureIndex() != null){
				
				Double featureValue = dataValue[node.getFeatureIndex()];
				// Check whether flow should go to left or right
				if(featureValue < node.getThresholdValue()){
					node = node.getLeftChildNode();
				}else{
					node = node.getRightChildNode();
				}
				
			}else{
				double actualTargetValue = dataValue[targetValueIndex];
				double predictedTargetValue = node.getLabelValue();
				//System.out.println("Actual value : "+ actualTargetValue 
				//	+ " Predicted value :"+predictedTargetValue);
				if(actualTargetValue == predictedTargetValue){
					return true;
				}
				else{
					return false;
				}
					
			}
		}
		
		
	}
	
	public static Double predictionValue(double dataValue[],Node rootNode){
		
		Node node = rootNode;
		
		while(true){

			if(node.getFeatureIndex() != null){
				
				Double featureValue = dataValue[node.getFeatureIndex()];
				// Check whether flow should go to left or right
				if(featureValue < node.getThresholdValue()){
					node = node.getLeftChildNode();
				}else{
					node = node.getRightChildNode();
				}
				
			}else{
				
				double predictedTargetValue = node.getLabelValue();
				return predictedTargetValue;
					
			}
		}
		
		
	}
	
	public Matrix getDataMatrix(){
		return dataMatrix;
	}
	
	public Double sumDataDistribution(List<Integer> dataPoints, HashMap<Integer,Double> dataDistribution){
		
		double sum = 0d;
		for(Integer dataPoint : dataPoints){
			
			sum += dataDistribution.get(dataPoint);
			
		}
		
		return sum;
		
	}


}
