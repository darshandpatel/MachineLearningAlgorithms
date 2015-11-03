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
public class DecisionTree{

	
	private FileOperations fileOperations;
	Matrix dataMatrix;
	HashMap<Integer,List<Integer>> dataPointPerFold;
	
	
	
	/**
	 * Constructor : Which fetches the Data Matrix from the source file
	 * and creates the Matrix from it.
	 */
	public DecisionTree(){
		
		// Create the mapping of DataPoint and its start byte code from the file 
		fileOperations =  new FileOperations();
		dataMatrix =  fileOperations.fetchDataPointsFromFile(
				Constant.SPAMBASE_DATA_FILE_PATH,Constant.SPAMBASE_DATA_FILE_NAME,
				Constant.NBR_OF_DP,Constant.NBR_OF_FEATURES + 1
				,Constant.COMMA_REGEX);
	}
	
	
	public void formDifferentDataPerFold(){
		
		dataPointPerFold = new HashMap<Integer,List<Integer>>();
		
		int numOfDPPerFold = Constant.NBR_OF_DP / Constant.NBR_OF_FOLDS;
		int extraDP = Constant.NBR_OF_DP % Constant.NBR_OF_FOLDS;
		
		for(int i = 0; i < Constant.NBR_OF_FOLDS; i++){
			
			List<Integer> numbers = IntStream.iterate(i, n -> n + 10).limit(numOfDPPerFold + ((i ==0)?extraDP:0))
					.boxed().collect(Collectors.toList());
			dataPointPerFold.put(i, numbers);
		}
		
	}
	
	
	public HashMap<String,Matrix> formDataMatrixByFold(Integer numOfFolds,Integer currentFoldCount){
		
		HashMap<String,Matrix> matrixHashMap = new HashMap<String,Matrix>();
		
		ArrayList<ArrayList<Double>> trainDataArrayList = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> testDataArrayList = new ArrayList<ArrayList<Double>>();
		
		for(int i = 0; i < Constant.NBR_OF_FOLDS; i++){
			
			if(i != currentFoldCount){
				
				List<Integer> dataPoints = dataPointPerFold.get(i);
				Iterator<Integer> dataPointIterator = dataPoints.iterator();
				
				while(dataPointIterator.hasNext()){
					
					int rowCount = dataPointIterator.next();
					ArrayList<Double> attribeValues = new ArrayList<Double>();
					for(int j = 0 ; j <= Constant.NBR_OF_FEATURES;j++)
						attribeValues.add(dataMatrix.get(rowCount, j));
					trainDataArrayList.add(attribeValues);
				}
				
			}else{
				
				List<Integer> dataPoints = dataPointPerFold.get(i);
				Iterator<Integer> dataPointIterator = dataPoints.iterator();
				
				while(dataPointIterator.hasNext()){
					
					int rowCount = dataPointIterator.next();
					ArrayList<Double> attribeValues = new ArrayList<Double>();
					for(int j = 0 ; j <= Constant.NBR_OF_FEATURES;j++)
						attribeValues.add(dataMatrix.get(rowCount, j));
					testDataArrayList.add(attribeValues);
				}
				
			}
			
		}
		
		// Convert the ArrayList into 2 D array
		double trainData[][] = new double[trainDataArrayList.size()][];
		double testData[][] = new double[testDataArrayList.size()][];
		
		int nbrOFTrainData = trainDataArrayList.size();
		int nbrOfTestData = testDataArrayList.size();
		
		for (int i = 0; i < nbrOFTrainData; i++) {
		    ArrayList<Double> row = trainDataArrayList.get(i);
		    trainData[i] = Doubles.toArray(row);
		}
		
		for (int i = 0; i < nbrOfTestData; i++) {
		    ArrayList<Double> row = trainDataArrayList.get(i);
		    testData[i] = Doubles.toArray(row);
		}
		
		matrixHashMap.put(Constant.TRAIN, new Matrix(trainData));
		matrixHashMap.put(Constant.TEST, new Matrix(testData));
		
		return matrixHashMap;
	}
	
	public HashMap<String,Object> calculateEntropy(ArrayList<Integer> dataPoints, 
			HashMap<Integer, Double> dataDistribution){
		
		double sumOfSpamDataDistribution = 0;
		double sumOfNonSpamDataDistribution = 0;
		int numOfSpamInstant = 0;
		int numOfNonSpamInstant = 0;
		
		HashMap<String,Object> entropyMap = new HashMap<String,Object>();
		//System.out.println("Matrix # of rows :"+dataMatrix.getRowDimension());
		//System.out.println("Matrix # of columns :"+dataMatrix.getColumnDimension());
		
		for(Integer dataPoint : dataPoints){
			if(dataMatrix.get(dataPoint,Constant.SPAMBASE_DATA_TARGET_VALUE_INDEX) == 1){
				sumOfSpamDataDistribution += dataDistribution.get(dataPoint);
				numOfSpamInstant++;
			}
			else{
				sumOfNonSpamDataDistribution += dataDistribution.get(dataPoint);
				numOfNonSpamInstant++;
			}
		}

		double spamProbability = sumOfSpamDataDistribution;
		double hamProbability = sumOfNonSpamDataDistribution;
		
		Double entropy = -((spamProbability * log2(spamProbability)) + 
				(hamProbability * log2(hamProbability)));
		
		//System.out.println("Entropy : "+entropy);
		
		entropyMap.put(Constant.ENTROPY, entropy);
		entropyMap.put(Constant.SPAM_COUNT, numOfSpamInstant);
		entropyMap.put(Constant.HAM_COUNT, numOfNonSpamInstant);
		
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
	
	
	public void formTrainDecisionTree(Node rootNode,Matrix trainDataMatrix, 
			HashMap<Integer, Double> dataDistribution){
		
		Integer exploredNodeCount = 0;
		Integer exploredNodeLimit = 1;
		for(int i = 0; i < Constant.SPAMBASE_DATA_DEPTH_LIMIT; i++){
			exploredNodeLimit += (int)Math.pow(2,Constant.SPAMBASE_DATA_DEPTH_LIMIT);
		}
		
		Queue<Node> nodeQueue = new LinkedList<Node>();
		nodeQueue.add(rootNode);
		
		// Form the tree until Node queue is not empty and the # of explored node 
		// is less the # of explored node limit
		
		while( (nodeQueue.size() > 0) && (exploredNodeCount < exploredNodeLimit)){
			
			Node currentNode = nodeQueue.poll();
			exploredNodeCount++;
		
			splitNodeByBestFeaturethreshold(currentNode, dataDistribution);
			
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
	
	public void splitNodeByBestFeaturethreshold(Node node, HashMap<Integer, Double> dataDistribution){
		
		Double parentNodeEntrpy = node.getEntropy();
		ArrayList<Integer> currentNodeDataPoints = node.getDataPoints();
		Integer noOfCurrentNodeDataPoints = currentNodeDataPoints.size();
		
		ArrayList<Feature> features = fetchFeaturePosThreshold(dataMatrix,currentNodeDataPoints);
		
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
					
					for(int k = 0 ; k < noOfCurrentNodeDataPoints ; k++){
						
						Integer dataPoint = currentNodeDataPoints.get(k);
						Double trainFeatureValue = dataMatrix.get(dataPoint,featureIndex);
						
						if(trainFeatureValue < thresholdValues.get(i)){
							
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
			}
			
			//TODO Only single method can return the best feature index
			// Calculation of square error (SE)
			
			HashMap<Integer,Double> entropyPerThresholdValue = calculateEntropy(feature,
					leftSideDPPerThreshold,rightSideDPPerThreshold,dataDistribution);
			
			
			Iterator<Entry<Integer, Double>> entropyIterator = 
					entropyPerThresholdValue.entrySet().iterator();
			Double lowestEntropy = Double.POSITIVE_INFINITY;
			Integer bestThresholdIndex=0;
			
			while(entropyIterator.hasNext()){
				Entry<Integer, Double> entry = entropyIterator.next();
				if(entry.getValue() < lowestEntropy){
					lowestEntropy = entry.getValue();
					bestThresholdIndex = entry.getKey();
				}
			}
			
			bestThresholdIndexPerFeature.put(featureIndex,bestThresholdIndex);
			
			Double infoGain = parentNodeEntrpy - (double)(lowestEntropy/currentNodeDataPoints.size());
			infoGainPerFeatureBestThreshold.put(featureIndex,infoGain);
			
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
		
		//System.out.println("*********************ALL FEATURES SCANNED************");
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
		Feature bestFeature = features.get(bestFeatureIndex);
		
		Integer bestThresholdValueIndex = bestThresholdIndexPerFeature.get(bestFeatureIndex);
		node.setThresholdValue(bestFeature.getThresholdValues().get(bestThresholdValueIndex));
		node.setFeatureIndex(bestFeatureIndex);
		
		
		//createChildNodes(node,features,infoGainPerFeatureBestThreshold, bestThresholdIndexPerFeature, 
		//		leftSideDPForFeaturesBestThreshold,rightSideDPForFeatureBestThreshold, dataDistribution);
		
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
				totalChildNodeEntroy += (entropyOfLeftChild * leftDataPoints.get(i).size());
			}
			//System.out.println("Left side entropy : "+entropyOfLeftChild);
			
			Double entropyRightChild = 0d;
			if(rightDataPoints.containsKey(i)){
				//System.out.println(rightDataPoints.get(i).toString());
				HashMap<String,Object> entropyRightSideMap = 
						calculateEntropy(rightDataPoints.get(i),
								normalizeDataDistribution(rightDataPoints.get(i),dataDistribution));
				entropyRightChild = (Double)entropyRightSideMap.get(Constant.ENTROPY);
				totalChildNodeEntroy += (entropyRightChild * rightDataPoints.get(i).size());
			}
			//System.out.println("Right side entropy : "+entropyRightChild);
			
			//System.out.println("Threshold Index : "+ i +"\t Threshold value : "+thresholdValues.get(i)
			//		+ "\t Error Value : "+error);
			entropyPerThreshold.put(i, totalChildNodeEntroy);
			
		}
		return entropyPerThreshold;
	}
	
	public HashMap<Integer, Double> normalizeDataDistribution(ArrayList<Integer> dataPoints,
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
	
	public ArrayList<Feature> fetchFeaturePosThreshold(Matrix dataMatrix, 
			ArrayList<Integer> dataPoints){
		
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
 
			List<Double> values = featurePosThreshold.get(i);
			double minValue = values.stream().mapToDouble(p -> p).min().getAsDouble();
			values.add(minValue - 1);
			
			double maxValue = values.stream().mapToDouble(p -> p).max().getAsDouble();
			values.add(maxValue + 1);
			
		}
		
		
		return createFeatures(featurePosThreshold);
		
	}
	
	/**
	 * 
	 * @param featurePosThreshold : A HashMap which has key as feature index and
	 * value as the possible threshold value
	 * @return The ArrayList of feature objects.
	 */
	private ArrayList<Feature> createFeatures(HashMap<Integer,List<Double>> 
			featuresPosThreshold){
	
		ArrayList<Feature> features = new ArrayList<Feature>();
		for(int i = 0; i < Constant.NBR_OF_FEATURES;i++){
			
			String featureCtg = null;
			
			List<Double> calculatedFeaturePosCriValues = featuresPosThreshold.get(i);
			Collections.sort(calculatedFeaturePosCriValues);
			
			featureCtg = Constant.NUMERIC;
			calculatedFeaturePosCriValues = filterFeaturePosThreshold
				(calculatedFeaturePosCriValues);
			
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
	
	
	/**
	 * This method creates the child nodes for the given node.
	 * 
	 * @param node 		: The node for which child nodes need to be created.
	 * @param features 	: The available features for the split
	 * @param infoGainPerFeatureBestThreshold 	: HashMap which contains the feature index
	 *  as key and the maximum information it can again after the split as value.
	 * @param bestThresholdIndexPerFeature		:  HashMap which contains the feature index
	 *  as key and the best threshold value for the feature as value.
	 * @param leftSideDPForFeaturesBestThreshold	: HashMap which contains the feature
	 * index as key and the Possible Data Points of the given node's left child if the node 
	 * is split by the feature.
	 * @param rightSideDPForFeatureBestThreshold	: HashMap which contains the feature
	 * index as key and the Possible Data Points of the given node's right child if the node 
	 * is split by the feature.
	 */
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
			
		if(leftSideDPForFeaturesBestThreshold.containsKey(bestFeatureIndex)){
			
			Node leftNode = new Node();
			//System.out.println("Left child # of data points "
			//				+leftSideDPForFeaturesBestThreshold.get(bestFeatureIndex).size());
			leftNode.setDataPoints(leftSideDPForFeaturesBestThreshold.get(bestFeatureIndex));
			//System.out.println(leftNode.getDataPoints().toString());
			leftNode.setParentNode(node);
			HashMap<String,Object> entropyMap = calculateEntropy(leftNode.getDataPoints(),
					normalizeDataDistribution(leftNode.getDataPoints(),dataDistribution));
			leftNode.setEntropy((Double)entropyMap.get(Constant.ENTROPY));
			leftNode.setLabelValue(
					((Integer)entropyMap.get(Constant.SPAM_COUNT) > 
					(Integer)entropyMap.get(Constant.HAM_COUNT))?1d:0d);
			//System.out.println("LEft label value :"+leftNode.getLabelValue());
			//System.out.println("LEft node entropy :"+(Double)entropyMap.get(Constant.ENTROPY));
			node.setLeftChildNode(leftNode);
			
		}
		
		if(rightSideDPForFeatureBestThreshold.containsKey(bestFeatureIndex)){
			
			Node rightNode = new Node();
			//System.out.println("Right child # of data points "
			//				+rightSideDPForFeatureBestThreshold.get(bestFeatureIndex).size());
			rightNode.setDataPoints(rightSideDPForFeatureBestThreshold.get(bestFeatureIndex));
			//System.out.println(rightNode.getDataPoints().toString());
			rightNode.setParentNode(node);
			
			HashMap<String,Object> entropyMap = 
					calculateEntropy(rightNode.getDataPoints(),
							normalizeDataDistribution(rightNode.getDataPoints(),dataDistribution));
			rightNode.setEntropy((Double)entropyMap.get(Constant.ENTROPY));
			rightNode.setLabelValue(
					((Integer)entropyMap.get(Constant.SPAM_COUNT) > 
					(Integer)entropyMap.get(Constant.HAM_COUNT))?1d:0d);
			//System.out.println("Right label value :"+rightNode.getLabelValue());
			//System.out.println("Right node entropy :"+(Double)entropyMap.get(Constant.ENTROPY));
			node.setRightChildNode(rightNode);
		}
			
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
	
	public HashMap<String, Object> calculateWeightedError(Matrix trainDataMatrix,Node rootNode,
			HashMap<Integer, Double> dataDistribution){
		
		HashMap<String, Object> returnObj = new HashMap<String, Object>();
		double trainDataArray[][] = trainDataMatrix.getArray();
		List<Integer> misClassifiedDataPoints = new ArrayList<Integer>();
		List<Integer> correctlyClassifiedDataPoints = new ArrayList<Integer>();
		
		Double error = 0d;
		Integer numOfDP = trainDataMatrix.getRowDimension();
		for(int i=0;i< numOfDP;i++){
			
			if(validatePredictionValue(trainDataArray[i],rootNode) == 0){
				
				error += dataDistribution.get(i);
				misClassifiedDataPoints.add(i);
				
			}
			else{
				correctlyClassifiedDataPoints.add(i);
			}
		}
		returnObj.put(Constant.ERROR_VALUE, error);
		returnObj.put(Constant.MISCLASSIFIED_DP, misClassifiedDataPoints);
		returnObj.put(Constant.CORRECTLY_CLASSIFIED_DP, correctlyClassifiedDataPoints);
		
		return returnObj;
		
	}
	
	/**
	 * This method evaluates the generated Decision Tree Model 
	 * based upon the Test Data set.
	 * 
	 * @param testDataMatrix 	: The test Data Matrix
	 * @param rootNode			: The root node of Decision Tree	
	 * @return The accuracy of the Decision tree based upon the prediction
	 * on the test Data Matrix.
	 */
	
	public Double evaluateTestDataSet(Matrix testDataMatrix,Node rootNode){
		
		double testDataArray[][] = testDataMatrix.getArray();
		
		Double score = 0d;
		Integer numOfDP = testDataMatrix.getRowDimension();
		for(int i=0;i< numOfDP;i++){
			score += validatePredictionValue(testDataArray[i],rootNode);
		}
		
		Double accuracy = score / testDataMatrix.getRowDimension();
		System.out.println("Match score :"+score);
		System.out.println("# of Records :"+testDataMatrix.getRowDimension());
		System.out.println("Accuracy :" + accuracy);
		return accuracy;
		
	}

	/**
	 * 
	 * @param dataValue : The single Data Point from the test Data Matrix.
	 * @param rootNode	: The root Node of the Decision Tree
	 * @return			: 1 if the given decision tree can predict the target
	 * value of the given data point correctly otherwise return 0
	 */
	public Integer validatePredictionValue(double dataValue[],Node rootNode){
		
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
				double actualTargetValue = dataValue[Constant.SPAMBASE_DATA_TARGET_VALUE_INDEX];
				double predictedTargetValue = node.getLabelValue();
				//System.out.println("Actual value : "+ actualTargetValue 
				//	+ " Predicted value :"+predictedTargetValue);
				if(actualTargetValue == predictedTargetValue){
					return 1;
				}
				else{
					return 0;
				}
					
			}
		}
		
		
	}

}
