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
	 * Constructor : Which fetches the Data Matrix from the train Data set file 
	 * and create the root node of Decision tree.
	 */
	public DecisionTree(){
		
		// Create the mapping of DataPoint and its start byte code from the file 
		fileOperations =  new FileOperations();
		dataMatrix =  fileOperations.fetchDataPointsFromFile(
				Constant.SPAMBASE_DATA_FILE_PATH,Constant.SPAMBASE_DATA_FILE_NAME,
				Constant.SPAMBASE_DATA_NUM_OF_DP,Constant.SPAMBASE_DATA_NUM_OF_FEATURES + 1
				,",");
		
	}
	
	public void formMultipleDecisionTrees(){
		
		int numOfFolds = 10;
		int currentFoldCount = 1;
		
		while(currentFoldCount <= numOfFolds){
			
			
			//System.out.println("Current Fold number" + currentFoldCount);
			
			// Form Decision Tree
			Matrix trainDataMatrix = formTrainDataMatrixByFold(numOfFolds,currentFoldCount);
			//System.out.println("Train Data Matrix Dimenstions");
			//System.out.printf("%s : %d\n%s : %d","# of rows",trainDataMatrix.getRowDimension(),
			//		"# od columns",trainDataMatrix.getColumnDimension());
			
			Node rootNode = new Node();
			rootNode.setDataPoints(dataMatrix.getRowDimension());
			HashMap<String,Object> entropyMap = calculateEntropy(rootNode.getDataPoints(),trainDataMatrix);
			rootNode.setEntropy((Double)entropyMap.get(Constant.ENTROPY));
			rootNode.setLabelValue(
					((Double)entropyMap.get(Constant.SPAM_COUNT) > 
					(Double)entropyMap.get(Constant.HAM_COUNT))?1d:0d);
			formTrainDecisionTree(rootNode,trainDataMatrix);
			
			Matrix testDataMatrix = formTestDataMatrixByFold(numOfFolds,currentFoldCount);
			System.out.println("Train Data Matrix Dimenstions");
			System.out.printf("%s : %d\n%s : %d","# of rows",testDataMatrix.getRowDimension(),
					"# od columns",testDataMatrix.getColumnDimension());
			
			currentFoldCount++;
			// Evaluate Decision Tree
			
			// Accumulate Results
			
		}
		
		// Print the Results
	}
	
	
	
	public Matrix formTrainDataMatrixByFold(Integer numOfFolds,Integer currentFoldCount){
		
		int numOfDPPerFold = Constant.SPAMBASE_DATA_NUM_OF_DP / numOfFolds;
		int extraDP = (currentFoldCount != numOfFolds?Constant.SPAMBASE_DATA_NUM_OF_DP % numOfFolds:0);
		
		int numOfDPInTrain = ((numOfFolds - 1) * numOfDPPerFold) + extraDP;
		double trainData[][] = 
				new double[numOfDPInTrain][Constant.SPAMBASE_DATA_NUM_OF_FEATURES];
		int trainDPIndex = 0;
		
		for(int i = 1; i <= numOfFolds; i++){
			
			if(i != currentFoldCount){
				
				int startIndex = (i - 1) * numOfDPPerFold;
				int endIndex = startIndex + (numOfDPPerFold - 1) + (currentFoldCount == numOfFolds?extraDP:0);
				
				while(startIndex <= endIndex){
					for(int j = 0 ; j <Constant.SPAMBASE_DATA_NUM_OF_FEATURES;j++)
						trainData[trainDPIndex][j] = dataMatrix.get(startIndex, j);
					startIndex++;
					trainDPIndex++;
				}
			}
		}
		
		return new Matrix(trainData);
	}
	
	public Matrix formTestDataMatrixByFold(Integer numOfFolds,Integer currentFoldCount){
		
		int numOfDPPerFold = Constant.SPAMBASE_DATA_NUM_OF_DP / numOfFolds;
		int extraDP = (currentFoldCount == numOfFolds?Constant.SPAMBASE_DATA_NUM_OF_DP % numOfFolds:0);
		int numOfDPInTrain = numOfDPPerFold + extraDP;
		double testData[][] = 
				new double[numOfDPInTrain][Constant.SPAMBASE_DATA_NUM_OF_FEATURES];
				
		int startIndex = (currentFoldCount - 1) * numOfDPPerFold;
		int endIndex = startIndex + (numOfDPPerFold - 1) + extraDP;
		int testDPIndex = 0;
		
		while(startIndex <= endIndex){
			for(int j = 0 ; j <Constant.SPAMBASE_DATA_NUM_OF_FEATURES;j++)
				testData[testDPIndex][j] = dataMatrix.get(startIndex, j);
			startIndex++;
			testDPIndex++;
		}
		
		return new Matrix(testData);
	}

	public HashMap<String,Object> calculateEntropy(ArrayList<Integer> dataPoints,Matrix dataMatrix){
		
		int numOfSpamInstant = 0;
		int numOfHamInstant = 0;
		HashMap<String,Object> entropyMap = new HashMap<String,Object>();
		
		for(Integer dataPoint : dataPoints){
			if(dataMatrix.get(dataPoint,Constant.SPAMBASE_DATA_TARGET_VALUE_INDEX) == 1)
				numOfSpamInstant++;
			else
				numOfHamInstant++;
		}
		
		int totalInstanct = numOfHamInstant + numOfSpamInstant;
		double spamProbability = (double)numOfSpamInstant/totalInstanct;
		double hamProbability = (double)numOfHamInstant/totalInstanct;
		
		Double entropy = -((spamProbability * log2(spamProbability)) + (hamProbability * log2(hamProbability)));
		
		entropyMap.put(Constant.ENTROPY, entropy);
		entropyMap.put(Constant.SPAM_COUNT, numOfSpamInstant);
		entropyMap.put(Constant.HAM_COUNT, numOfHamInstant);
		
		return entropyMap;
	}
	
	private static double log2(double n)
	{
	    return (Math.log(n) / Math.log(2));
	}
	
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
			//System.out.println("Parent node error is 		:"+currentNode.getError());
			
			splitNodeByBestFeaturethreshold(currentNode);
			
			if(currentNode.getLeftChildNode() != null){
				
				Node leftNode = currentNode.getLeftChildNode();
				if(leftNode.getError() > Constant.HOUSING_DATA_ERROR_THRESHOLD)
					nodeQueue.add(currentNode.getLeftChildNode());
				
			}
			if(currentNode.getRightChildNode() != null){
				
				Node rightNode = currentNode.getRightChildNode();
				if(rightNode.getError() > Constant.HOUSING_DATA_ERROR_THRESHOLD)
					nodeQueue.add(currentNode.getRightChildNode());
				
			}
		}
		
	}
	
	
	/**
	 * This method find the best feature and its best threshold value to
	 * split the given Regression tree node.
	 * If the split is possible then the method will create the child nodes
	 * of the given node.
	 * @param node : The node
	 */
	public void splitNodeByBestFeaturethreshold(Node node){
		
		Double parentNodeEntrpy = node.getEntropy();
		ArrayList<Integer> currentNodeDataPoints = node.getDataPoints();
		Integer NoOfCurrentNodeDataPoints = currentNodeDataPoints.size();
		
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
			//System.out.println("Feature index"+feature.getIndex());
			//System.out.println("Feature Threshold" + thresholdValues.toString());
			int noOfThresholdValues = thresholdValues.size();
			
			if(feature.getType().equals(Constant.NUMERIC)){
				
				for(int i = 0 ; i < noOfThresholdValues;i++){
					
					for(int k = 0 ; k < NoOfCurrentNodeDataPoints ; k++){
						
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
								//System.out.println("Right side node Threshol value index : "+i+"
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
					
					for(int k = 0 ; k < NoOfCurrentNodeDataPoints ; k++){
					
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
			
			
			Iterator<Entry<Integer, Double>> labelSEIterator = entropyPerThresholdValue.entrySet().iterator();
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
			
			Double infoGain = parentNodeEntrpy - (lowestEntropy/currentNodeDataPoints.size());
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
				//leftSideDPPerThreshold.get(bestThresholdIndex).size());
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
	
	public HashMap<Integer,Double> calculateEntropy(Feature feature,
			HashMap<Integer,ArrayList<Integer>> leftDataPoints,
			HashMap<Integer,ArrayList<Integer>> rightDataPoints){
			
		HashMap<Integer,Double> entropyPerThreshold = new HashMap<Integer,Double>();
		ArrayList<Double> thresholdValues = feature.getThresholdValues();
		
		int noOfThresholdValues = thresholdValues.size();
		
		for(int i =0 ; i < noOfThresholdValues; i++){
			
			Double entropyOfLeftChild = 0d;
			if(leftDataPoints.containsKey(i)){
				//System.out.println(leftDataPoints.get(i).toString());
				HashMap<String,Object> entropyLeftSideMap= calculateEntropy(leftDataPoints.get(i),dataMatrix);
				entropyOfLeftChild = (Double)entropyLeftSideMap.get(Constant.ENTROPY);
			}
			//System.out.println("Left side error : "+leftLabelValueError);
			
			Double entropyRightChild = 0d;
			if(rightDataPoints.containsKey(i)){
				//System.out.println(rightDataPoints.get(i).toString());
				HashMap<String,Object> entropyRightSideMap= calculateEntropy(rightDataPoints.get(i),dataMatrix);
				entropyRightChild = (Double)entropyRightSideMap.get(Constant.ENTROPY);
			}
			//System.out.println("Right side error : "+rightLabelValueError);
			
			Double totalChildNodeEntroy = ((leftDataPoints.get(i).size()*entropyOfLeftChild) + 
					(rightDataPoints.get(i).size()*entropyRightChild));
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
	
	private ArrayList<Feature> createFeatures(HashMap<Integer , 
			ArrayList<Double>> featurePosThreshold){
	
		ArrayList<Feature> features = new ArrayList<Feature>();
		for(int i = 0; i < Constant.SPAMBASE_DATA_NUM_OF_FEATURES;i++){
			
			String featureCtg = null;
			
			ArrayList<Double> calculatedFeaturePosCriValues = featurePosThreshold.get(i);
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
			node.setFeatureIndex(bestFeature.getIndex());
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
						((Double)entropyMap.get(Constant.SPAM_COUNT) > 
						(Double)entropyMap.get(Constant.HAM_COUNT))?1d:0d);
				//System.out.println("LEft label value :"+leftNode.getLabelValue());
				//System.out.println("LEft node error :"+leftNode.getError());
				node.setLeftChildNode(leftNode);
				
			}
			
			if(rightSideDPForFeatureBestThreshold.containsKey(bestFeatureIndex)){
				
				Node rightNode = new Node();
				//System.out.println("Right child # of data points "
				//				+rightSideDPForFeatureBestThreshold.get(bestFeatureIndex).size());
				rightNode.setDataPoints(rightSideDPForFeatureBestThreshold.get(bestFeatureIndex));
				//System.out.println(rightNode.getDataPoints().toString());
				rightNode.setParentNode(node);
				
				HashMap<String,Object> entropyMap = calculateEntropy(rightNode.getDataPoints(), dataMatrix);
				rightNode.setEntropy((Double)entropyMap.get(Constant.ENTROPY));
				rightNode.setLabelValue(
						((Double)entropyMap.get(Constant.SPAM_COUNT) > 
						(Double)entropyMap.get(Constant.HAM_COUNT))?1d:0d);
				//System.out.println("Right label value :"+rightNode.getLabelValue());
				//System.out.println("Right node error :"+rightNode.getError());
				node.setRightChildNode(rightNode);
			}
			
		}
		
	}

}
