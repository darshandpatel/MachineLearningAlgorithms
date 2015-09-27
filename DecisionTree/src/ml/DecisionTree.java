package ml;

import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Queue;
import java.util.ArrayList;
import java.util.Map.Entry;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import Jama.Matrix;

/**
 * 
 * @author Darshan
 *
 */
public class DecisionTree{

	
	private FileOperations fileOperations;
	Matrix dataMatrix;
	
	
	/**
	 * Constructor : Which fetches the Data Matrix from the source file
	 * and creates the Matrix from it.
	 */
	public DecisionTree(){
		
		// Create the mapping of DataPoint and its start byte code from the file 
		fileOperations =  new FileOperations();
		dataMatrix =  fileOperations.fetchDataPointsFromFile(
				Constant.SPAMBASE_DATA_FILE_PATH,Constant.SPAMBASE_DATA_FILE_NAME,
				Constant.SPAMBASE_DATA_NUM_OF_DP,Constant.SPAMBASE_DATA_NUM_OF_FEATURES + 1
				,",");
		
	}
	
	/**
	 * This method creates the Multiple Decision Tree by dividing 
	 * the given DataSet into different training and testing portion.
	 */
	public void formMultipleDecisionTrees(){
		
		// Number of fold will be created on the given Data Matrix;
		int numOfFolds = 10;
		int currentFoldCount = 1;
		
		// Total average accuracy of all the Decision Trees.
		Double sumOfAccuracy = 0d;
		
		while(currentFoldCount <= numOfFolds){
			
			//System.out.println("Current Fold number" + currentFoldCount);
			
			// Form Decision Tree
			System.out.println("===============================");
			System.out.println("Fold count : "+currentFoldCount);
			
			HashMap<String,Matrix> matrixHashMap = formDataMatrixByFold(numOfFolds,currentFoldCount);
			Matrix trainDataMatrix = matrixHashMap.get(Constant.TRAIN);
			
			//System.out.println("Train Data Matrix Dimensions");
			//System.out.printf("%s : %d\n%s : %d","# of rows",trainDataMatrix.getRowDimension(),
			//		"# od columns",trainDataMatrix.getColumnDimension());
			
			// Root Node for the current Decision Tree
			Node rootNode = new Node();
			rootNode.setDataPoints(trainDataMatrix.getRowDimension());
			HashMap<String,Object> entropyMap = calculateEntropy(rootNode.getDataPoints(),trainDataMatrix);
			
			rootNode.setEntropy((Double)entropyMap.get(Constant.ENTROPY));
			rootNode.setLabelValue(
					((Integer)entropyMap.get(Constant.SPAM_COUNT) > 
					(Integer)entropyMap.get(Constant.HAM_COUNT))?1d:0d);
			formTrainDecisionTree(rootNode,trainDataMatrix);
			//System.out.println("Print the Decision Tree");
			
			printDecisionTree(rootNode);

			// Evaluate Decision Tree
			Matrix testDataMatrix = matrixHashMap.get(Constant.TEST);
			//System.out.println("Train Data Matrix Dimenstions");
			//System.out.printf("%s : %d\n%s : %d","# of rows\n",testDataMatrix.getRowDimension(),
			//		"# od columns",testDataMatrix.getColumnDimension());
			currentFoldCount++;
			
			// Accumulate Results
			Double accuracy = evaluateTestDataSet(testDataMatrix, rootNode);
			sumOfAccuracy += accuracy;
			//break;
			
		}
		
		// Print the Results
		System.out.println("Final accuracy : "+ sumOfAccuracy/numOfFolds);
		
	}
	
	
	/**
	 * This method split the whole Data Matrix into two parts, training and testing parts,
	 * based upon the given current fold count and total number of folds.
	 * 
	 * @param numOfFolds : Total number of the fold
	 * @param currentFoldCount : Current fold count
	 * @return : The HashMap which contains the training and testing Data Matrix based
	 * upon the given current fold count and number of the fold
	 */
	public HashMap<String,Matrix> formDataMatrixByFold(Integer numOfFolds,Integer currentFoldCount){
		
		HashMap<String,Matrix> matrixHashMap = new HashMap<String,Matrix>();
		
		int numOfDPPerFold = Constant.SPAMBASE_DATA_NUM_OF_DP / numOfFolds;
		
		// Extra Data Points
		int trainExtraDP = (currentFoldCount != numOfFolds?Constant.SPAMBASE_DATA_NUM_OF_DP % numOfFolds:0);
		int testExtraDP = (currentFoldCount == numOfFolds?Constant.SPAMBASE_DATA_NUM_OF_DP % numOfFolds:0);
		
		int numOfDPInTrain = ((numOfFolds - 1) * numOfDPPerFold) + trainExtraDP;
		int numOfDPInTest = numOfDPPerFold + testExtraDP;
		
		double trainData[][] = 
				new double[numOfDPInTrain][Constant.SPAMBASE_DATA_NUM_OF_FEATURES+1];
		double testData[][] = 
				new double[numOfDPInTest][Constant.SPAMBASE_DATA_NUM_OF_FEATURES+1];
		
		int trainDPIndex = 0;
		int testDPIndex = 0;
		
		for(int i = 1; i <= numOfFolds; i++){
			
			int startIndex = (i - 1) * numOfDPPerFold;
			
			if(i != currentFoldCount){
				int endIndex = startIndex + (numOfDPPerFold - 1) + 
						(currentFoldCount == numOfFolds?trainExtraDP:0);
				while(startIndex <= endIndex){
					for(int j = 0 ; j <= Constant.SPAMBASE_DATA_NUM_OF_FEATURES;j++)
						trainData[trainDPIndex][j] = dataMatrix.get(startIndex, j);
					startIndex++;
					trainDPIndex++;
				}
			}else{
				int endIndex = startIndex + (numOfDPPerFold - 1) + 
						(currentFoldCount == numOfFolds?testExtraDP:0);
				while(startIndex <= endIndex){
					for(int j = 0 ; j <= Constant.SPAMBASE_DATA_NUM_OF_FEATURES;j++)
						testData[testDPIndex][j] = dataMatrix.get(startIndex, j);
					startIndex++;
					testDPIndex++;
				}
				
			}
		}
		
		matrixHashMap.put(Constant.TRAIN, new Matrix(trainData));
		matrixHashMap.put(Constant.TEST, new Matrix(testData));
		return matrixHashMap;
	}
	

	/**
	 * 
	 * @param dataPoints : The Data Points
	 * @param dataMatrix : The Data Matrix
	 * @return : The entropy in the given Data points based upon the given Data Matrix.
	 */
	public HashMap<String,Object> calculateEntropy(ArrayList<Integer> dataPoints,Matrix dataMatrix){
		
		int numOfSpamInstant = 0;
		int numOfHamInstant = 0;
		
		HashMap<String,Object> entropyMap = new HashMap<String,Object>();
		//System.out.println("Matrix # of rows :"+dataMatrix.getRowDimension());
		//System.out.println("Matrix # of columns :"+dataMatrix.getColumnDimension());
		for(Integer dataPoint : dataPoints){
			if(dataMatrix.get(dataPoint,Constant.SPAMBASE_DATA_TARGET_VALUE_INDEX) == 1)
				numOfSpamInstant++;
			else
				numOfHamInstant++;
		}
		
		//System.out.println("# of spam : "+numOfSpamInstant);
		//System.out.println("# of ham : "+numOfHamInstant);
		
		int totalInstanct = numOfHamInstant + numOfSpamInstant;
		double spamProbability = (double)numOfSpamInstant/totalInstanct;
		double hamProbability = (double)numOfHamInstant/totalInstanct;
		
		Double entropy = -((spamProbability * log2(spamProbability)) + 
				(hamProbability * log2(hamProbability)));
		
		//System.out.println("Entropy : "+entropy);
		
		entropyMap.put(Constant.ENTROPY, entropy);
		entropyMap.put(Constant.SPAM_COUNT, numOfSpamInstant);
		entropyMap.put(Constant.HAM_COUNT, numOfHamInstant);
		
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
	
	/**
	 * This method forms the Decision Tree based on the given training Data Matrix
	 * @param rootNode : The Root node of the Decision Tree which will be fully formed
	 * by this method
	 * @param trainDataMatrix : The Data Matrix of the Training Data Points.
	 */
	public void formTrainDecisionTree(Node rootNode,Matrix trainDataMatrix){
		
		Integer exploredNodeCount = 0;
		Integer exploredNodeLimit = (1 + (int)Math.pow(2,Constant.SPAMBASE_DATA_DEPTH_LIMIT));
		
		Queue<Node> nodeQueue = new LinkedList<Node>();
		nodeQueue.add(rootNode);
		
		// Form the tree until Node queue is not empty and the # of explored node 
		// is less the # of explored node limit
		
		while( (nodeQueue.size() > 0) && (exploredNodeCount < exploredNodeLimit)){
			
			Node currentNode = nodeQueue.poll();
			exploredNodeCount++;
			//System.out.println("##############################");
			//System.out.println("Parent node Data Points are	:" + 
			//currentNode.getDataPoints().size());
			//System.out.println("Parent node entropy is 		:"+currentNode.getEntropy());
			
		
			splitNodeByBestFeaturethreshold(currentNode);
			
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
	
	
	/**
	 * This method find the best feature and its best threshold value to
	 * split the given Decision tree node.
	 * If the split is possible then the method will create the child nodes
	 * of the given node.
	 * @param node : The node for which best split feature and threshold value
	 * needs to be found.
	 */
	public void splitNodeByBestFeaturethreshold(Node node){
		
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

			ArrayList<Double> thresholdValues = feature.getThresholdValues();
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
					leftSideDPPerThreshold,rightSideDPPerThreshold);
			
			
			Iterator<Entry<Integer, Double>> labelSEIterator = 
					entropyPerThresholdValue.entrySet().iterator();
			Double lowestEntropy = Double.POSITIVE_INFINITY;
			Integer bestThresholdIndex=0;
			while(labelSEIterator.hasNext()){
				Entry<Integer, Double> entry = labelSEIterator.next();
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
		
		createChildNodes(node,features,infoGainPerFeatureBestThreshold, bestThresholdIndexPerFeature, 
				leftSideDPForFeaturesBestThreshold,rightSideDPForFeatureBestThreshold);
		
	}
	
	/**
	 * 
	 * @param feature : A feature object
	 * @param leftDataPoints : HashMap which has possible Threshold value as key and
	 * the Left side data points after the split of a node, by the threshold value, as value.
	 * @param rightDataPoints : HashMap which has possible Threshold value as key and
	 * the right side data points after the split of a node, by the threshold value, as value.
	 * @return : HashMap which has key as the feature index and value as the entropy of the 
	 * right and left side data Points.
	 */
	public HashMap<Integer,Double> calculateEntropy(Feature feature,
			HashMap<Integer,ArrayList<Integer>> leftDataPoints,
			HashMap<Integer,ArrayList<Integer>> rightDataPoints){
			
		HashMap<Integer,Double> entropyPerThreshold = new HashMap<Integer,Double>();
		ArrayList<Double> thresholdValues = feature.getThresholdValues();
		
		int noOfThresholdValues = thresholdValues.size();
		
		for(int i =0 ; i < noOfThresholdValues; i++){
			
			Double totalChildNodeEntroy = 0d;
			Double entropyOfLeftChild = 0d;
			if(leftDataPoints.containsKey(i)){
				//System.out.println(leftDataPoints.get(i).toString());
				HashMap<String,Object> entropyLeftSideMap = 
						calculateEntropy(leftDataPoints.get(i),dataMatrix);
				entropyOfLeftChild = (Double)entropyLeftSideMap.get(Constant.ENTROPY);
				totalChildNodeEntroy += (entropyOfLeftChild * leftDataPoints.get(i).size());
			}
			//System.out.println("Left side entropy : "+entropyOfLeftChild);
			
			Double entropyRightChild = 0d;
			if(rightDataPoints.containsKey(i)){
				//System.out.println(rightDataPoints.get(i).toString());
				HashMap<String,Object> entropyRightSideMap = 
						calculateEntropy(rightDataPoints.get(i),dataMatrix);
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
	
	
	/**
	 * 
	 * @param dataMatrix : The Matrix which contains the training DataSet 
	 * @return ArrayList of Feature, Basically this method parse the dataPoints 
	 * matrix and find the possible threshold value for all the features.
	 */
	public ArrayList<Feature> fetchFeaturePosThreshold(Matrix dataMatrix, 
			ArrayList<Integer> dataPoints){
		
		int noOfColumns = dataMatrix.getColumnDimension();
		
		HashMap<Integer,ArrayList<Double>> featurePosThreshold = 
				new HashMap<Integer,ArrayList<Double>>(Constant.SPAMBASE_DATA_NUM_OF_FEATURES);
			
		for(int i=0; i < (noOfColumns -1);i++){
			
			for(Integer row : dataPoints){
				
				//System.out.println("value of i is : " + i+ " : "+parts[i]);
				//String featureName = Constant.features.get(i);
				if(featurePosThreshold.containsKey(i)){
					Double value = dataMatrix.get(row,i);
					ArrayList<Double> values = featurePosThreshold.get(i);
					if(!values.contains(value))
						values.add(value);
					
				}else{
					
					ArrayList<Double> values = new ArrayList<Double>();
					values.add(dataMatrix.get(row,i));
					featurePosThreshold.put(i,values);
				}
			}
		}
		
		return createFeatures(featurePosThreshold);
		
	}
	
	/**
	 * 
	 * @param featurePosThreshold : A HashMap which has key as feature index and
	 * value as the possible threshold value
	 * @return The ArrayList of feature objects.
	 */
	private ArrayList<Feature> createFeatures(HashMap<Integer,ArrayList<Double>> 
			featuresPosThreshold){
	
		ArrayList<Feature> features = new ArrayList<Feature>();
		for(int i = 0; i < Constant.SPAMBASE_DATA_NUM_OF_FEATURES;i++){
			
			String featureCtg = null;
			
			ArrayList<Double> calculatedFeaturePosCriValues = featuresPosThreshold.get(i);
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
	public ArrayList<Double> filterFeaturePosThreshold 
				(ArrayList<Double> calculatedFeaturePosThresholdValues){
		
		
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
			HashMap<Integer,ArrayList<Integer>> rightSideDPForFeatureBestThreshold){
		
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
		
		if(higherstInfoGain > Constant.SPAMBASE_DATA_INFO_GAIN_THRESHOLD){
			
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
				HashMap<String,Object> entropyMap = calculateEntropy(leftNode.getDataPoints(), dataMatrix);
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
						calculateEntropy(rightNode.getDataPoints(), dataMatrix);
				rightNode.setEntropy((Double)entropyMap.get(Constant.ENTROPY));
				rightNode.setLabelValue(
						((Integer)entropyMap.get(Constant.SPAM_COUNT) > 
						(Integer)entropyMap.get(Constant.HAM_COUNT))?1d:0d);
				//System.out.println("Right label value :"+rightNode.getLabelValue());
				//System.out.println("Right node entropy :"+(Double)entropyMap.get(Constant.ENTROPY));
				node.setRightChildNode(rightNode);
			}
			
		}
		
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
