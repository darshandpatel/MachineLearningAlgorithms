package ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map.Entry;
import java.util.Queue;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import Jama.Matrix;

/**
 * 
 * @author Darshan
 *
 */
public class BasicRegressionTree {
	
	private Queue<Node> nodeQueue;
	private Node rootNode;
	private FileOperations fileOperations;
	private Matrix dataMatrix;
	
	
	/**
	 * 
	 * @return The Root node of Regression Tree
	 */
	Node getRootNode(){
		return rootNode;
	}
	
	/**
	 * Constructor : Which fetches the Data Matrix from the train Data set file 
	 * and create the root node of regression tree.
	 */
	public BasicRegressionTree(){
		
		// Create node queue
		nodeQueue = new LinkedList<Node>();
		
		// Create the mapping of DataPoint and its start byte code from the file 
		fileOperations =  new FileOperations();
		dataMatrix =  fileOperations.fetchDataPointsFromFile(
				Constant.HOUSING_DATA_FILE_PATH,Constant.HOUSING_TRAINDATA_FILE,
				Constant.HOUSING_DATA_NUM_OF_TRAINING_DP,Constant.HOUSING_DATA_NO_OF_FEATURES+1,
				Constant.STRING_REGEX);
		
		// Create the root node for Regression Tree and add into Queue
		Node rootNode = new Node();
		rootNode.setDataPoints(dataMatrix.getRowDimension());
		rootNode.setError(calculateSumOfSquareError(rootNode.getDataPoints()));
		rootNode.setLabelValue(calculateMean(rootNode.getDataPoints()));
		
		this.rootNode = rootNode;
		nodeQueue.add(rootNode);
		
	}
	
	/**
	 * This method forms the regression tree on the given train data set.
	 */
	public void formRegressionTree(){
		
		Integer exploredNodeCount = 0;
		Integer exploredNodeLimit = 1;
		for(int i = 0; i < Constant.SPAMBASE_DATA_DEPTH_LIMIT; i++){
			exploredNodeLimit += (int)Math.pow(2,i);
		}
		
		// Form the tree until Node queue is not empty and the # of explored node 
		// is less the # of explored node limit
		while( (nodeQueue.size() > 0) && (exploredNodeCount < exploredNodeLimit)){
			
			Node currentNode = nodeQueue.poll();
			exploredNodeCount++;
			//System.out.println("##############################");
			//System.out.println("Parent node Data Points are	:" + 
			//currentNode.getDataPoints().size());
			System.out.println("Parent node error is 		:"+currentNode.getError());
			
			splitNodeByBestFeaturethreshold(currentNode);
			
			if(currentNode.getLeftChildNode() != null){
				
				Node leftNode = currentNode.getLeftChildNode();
				if( (leftNode.getError() > Constant.HOUSING_DATA_ERROR_THRESHOLD) &&
						leftNode.getDataPoints().size() > Constant.HOUSING_DATA_NUM_OF_CHILD_THRESHOLD)
					nodeQueue.add(leftNode);
			}
			if(currentNode.getRightChildNode() != null){
				
				Node rightNode = currentNode.getRightChildNode();
				if((rightNode.getError() > Constant.HOUSING_DATA_ERROR_THRESHOLD) &&
						rightNode.getDataPoints().size() > Constant.HOUSING_DATA_NUM_OF_CHILD_THRESHOLD)
					nodeQueue.add(rightNode);
			}
		}
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
				new HashMap<Integer,ArrayList<Double>>(Constant.HOUSING_DATA_NO_OF_FEATURES);
			
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
		for(int i = 0; i < Constant.HOUSING_DATA_NO_OF_FEATURES;i++){
			
			String featureName = Constant.HOUSING_DATA_FEATURUES.get(i);
			String featureCtg = null;
			
			ArrayList<Double> calculatedFeaturePosCriValues = featurePosThreshold.get(i);
			Collections.sort(calculatedFeaturePosCriValues);
			
			if(featureName.equals("CHAS")){
				featureCtg = Constant.BINARY_NUM;
			}else{
				featureCtg = Constant.NUMERIC;
				calculatedFeaturePosCriValues = filterFeaturePosThreshold
					(calculatedFeaturePosCriValues);
			}
			
			Feature feature = new Feature(featureName,featureCtg,calculatedFeaturePosCriValues,i);
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
	 * This method find the best feature and its best threshold value to
	 * split the given Regression tree node.
	 * If the split is possible then the method will create the child nodes
	 * of the given node.
	 * @param node : The node
	 */
	public void splitNodeByBestFeaturethreshold(Node node){
		
		Double parentNodeError = node.getError();
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
			HashMap<Integer,Double> labelSEPerThresholdValue = calculateLabelValueSE(feature,
					leftSideDPPerThreshold,rightSideDPPerThreshold);
			
			
			Iterator<Entry<Integer, Double>> labelSEIterator = labelSEPerThresholdValue.entrySet().iterator();
			Double lowestLabelSE = Double.POSITIVE_INFINITY;
			Integer bestThresholdIndex=0;
			while(labelSEIterator.hasNext()){
				Entry<Integer, Double> entry = labelSEIterator.next();
				if(entry.getValue() < lowestLabelSE){
					lowestLabelSE = entry.getValue();
					bestThresholdIndex = entry.getKey();
				}
			}
			
			bestThresholdIndexPerFeature.put(featureIndex,bestThresholdIndex);
			
			Double infoGain = parentNodeError - lowestLabelSE;
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
	
	
	/**
	 * This method creates the left and right child of the given node
	 * if the split information gain is more than the Minimum Information Gain.
	 * 
	 * @param node		: The Parent for which Child node should be created
	 * @param features	: The available features/features threshold values at
	 *  the parent nodes.
	 * @param infoGainPerFeatureBestThreshold 	: The HashMap which has key as the
	 * feature index and value as the Information Gain by the best available split by the feature
	 * @param bestThresholdIndexPerFeature		: The HashMap which has key as the
	 * feature index and value as the best threshold value index for which it has
	 * best Data set split
	 * @param leftSideDPForFeaturesBestThreshold : The HashMap which has key as 
	 * the feature index and value as the Data Points which falls left side after 
	 * the split of Data Set by the feature
	 * @param rightSideDPForFeatureBestThreshold : The HashMap which has key as 
	 * the feature index and value as the Data Points which falls right side after 
	 * the split of Data Set by the feature
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
		
		if(higherstInfoGain > Constant.HOUSING_DATA_INFO_GAIN_THRESHOLD){
			
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
				
				leftNode.setError(calculateSumOfSquareError(leftNode.getDataPoints()));
				leftNode.setLabelValue(calculateMean(leftNode.getDataPoints()));
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
				
				rightNode.setError(calculateSumOfSquareError(rightNode.getDataPoints()));
				rightNode.setLabelValue(calculateMean(rightNode.getDataPoints()));
				//System.out.println("Right label value :"+rightNode.getLabelValue());
				//System.out.println("Right node error :"+rightNode.getError());
				node.setRightChildNode(rightNode);
			}
			
		}
		
	}
	

	/**
	 * 
	 * @param feature 			: The feature
	 * @param leftDataPoints 	: The left side Data Points after 
	 * the split of the parent node by the given feature
	 * @param rightDataPoints	: The right side Data Points after the
	 * split of the parent node by the given feature 
	 * @return The Standard Error from the split of the Data Points
	 * for the given feature and its all threshold values.
	 */
	public HashMap<Integer,Double> calculateLabelValueSE(
			Feature feature,
			HashMap<Integer,ArrayList<Integer>> leftDataPoints,
			HashMap<Integer,ArrayList<Integer>> rightDataPoints){
			
		HashMap<Integer,Double> errorPerThreshold = new HashMap<Integer,Double>();
		ArrayList<Double> thresholdValues = feature.getThresholdValues();
		
		int noOfThresholdValues = thresholdValues.size();
		
		for(int i =0 ; i < noOfThresholdValues; i++){
			
			Double leftLabelValueError = 0d;
			if(leftDataPoints.containsKey(i)){
				//System.out.println(leftDataPoints.get(i).toString());
				leftLabelValueError = calculateSumOfSquareError(leftDataPoints.get(i));
			}
			//System.out.println("Left side error : "+leftLabelValueError);
			
			Double rightLabelValueError = 0d;
			if(rightDataPoints.containsKey(i)){
				//System.out.println(rightDataPoints.get(i).toString());
				rightLabelValueError = calculateSumOfSquareError(rightDataPoints.get(i));
			}
			//System.out.println("Right side error : "+rightLabelValueError);
			
			Double error = leftLabelValueError + rightLabelValueError;
			//System.out.println("Threshold Index : "+ i +"\t Threshold value : "+thresholdValues.get(i) 
			//		+ "\t Error Value : "+error);
			errorPerThreshold.put(i, error);
			
		}
		return errorPerThreshold;
	}
	
	
	/**
	 * 
	 * @param dataPoints : The matrix of Data Points
	 * @return The Error (Variance * # of records) of the target value from
	 * the given Data Point matrix
	 */
	public Double calculateError(Matrix dataPoints){
		
		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		for(int i=0;i<dataPoints.getRowDimension();i++){
			descriptiveStatistics.addValue(dataMatrix.get(i,Constant.HOUSING_DATA_TARGET_VALUE_INDEX));
		}
		return descriptiveStatistics.getVariance() * descriptiveStatistics.getN();
		
	}
	
	/**
	 * 
	 * @param values : Data Points label value ArrayList
	 * @return The Error (Variance * # of records) of the target value from
	 * the given Data Point ArrayList
	 */
	public Double calculateSumOfSquareError(ArrayList<Integer> dataPoints){
		
		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		for(Integer row : dataPoints){
			descriptiveStatistics.addValue(dataMatrix.get(row, Constant.HOUSING_DATA_TARGET_VALUE_INDEX));
		}
		return (descriptiveStatistics.getVariance() * descriptiveStatistics.getN());
	}
	
	
	
	/**
	 * 
	 * @param dataPoints : The ArrayList of all Data Point's Index (row/line number)
	 * @return The mean value of all Data Point's label value
	 */
	public Double calculateMean(ArrayList<Integer> dataPoints){
		
		if(dataPoints.size() > 0){
			DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
			for(Integer row : dataPoints){
				descriptiveStatistics.addValue(
						dataMatrix.get(row, Constant.HOUSING_DATA_TARGET_VALUE_INDEX));
			}
			return descriptiveStatistics.getMean();
		}else{
			return 0d;
		}
		
	}
	
	
	/**
	 * 
	 * @param dataPoints : Matrix of Data Points
	 * @return The mean of the label value of the given 
	 * Data Points
	 */
	public Double calculateMean(Matrix dataPoints){
		
		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		for(int i=0;i<dataPoints.getRowDimension();i++){
			descriptiveStatistics.addValue(
					dataMatrix.get(i,Constant.HOUSING_DATA_TARGET_VALUE_INDEX));
		}
		return descriptiveStatistics.getMean();
	}

	/**
	 * This method print the important information of the given Regression Tree
	 * Node
	 * @param node : Regression Tree Root Node
	 */
	public void printRegressionTree(){
		
		Queue<Node> nodeQueue =  new LinkedList<Node>();
		nodeQueue.add(this.rootNode);
		Integer count = 1;
		while(nodeQueue.size() > 0){
			
			Node node = nodeQueue.poll();
			
			System.out.printf("%30s\t\t\t:\t%d\n","Node number",count);
			System.out.printf("%30s\t\t\t:\t%f\n","Node Error",node.getError());
			System.out.printf("%30s\t\t\t:\t%d\n","Feature index",node.getFeatureIndex());
			System.out.printf("%30s\t\t\t:\t%f\n","Feature Threshold Value",node.getThresholdValue());
			System.out.printf("%30s\t\t\t:\t%f\n","Label Value",node.getLabelValue());
			System.out.printf("%30s\t\t\t:\t%f\n","Parent Node number",Math.floor(count/2));
			System.out.printf("%30s\t\t\t:\t%d\n\n","Number of Data Points",node.getDataPoints().size());
			
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
	 * This method evaluates the generated Regression Tree Model 
	 * based upon the Test Dataset.
	 * 
	 * It calculates the Mean Standard Error as the measurement of 
	 * accuracy. 
	 */
	public void evaluateTestDataSet(){
		
		Matrix dataMatrix =  fileOperations.fetchDataPointsFromFile(
				Constant.HOUSING_DATA_FILE_PATH,Constant.HOUSING_TESTDATA_FILE,
				Constant.HOUSING_DATA_NUM_OF_TESTING_DP,Constant.HOUSING_DATA_NO_OF_FEATURES+1,
				Constant.STRING_REGEX);
		double dataArray[][] = dataMatrix.getArray();
		
		Double standardError = 0d;
		for(int i=0;i< Constant.HOUSING_DATA_NUM_OF_TESTING_DP;i++){
			standardError += calculateSEOfPredictedValue(dataArray[i]);
		}
		
		System.out.println("Mean Standard Error :" + (standardError/
				Constant.HOUSING_DATA_NUM_OF_TESTING_DP));
		
	}
	
	/**
	 * 
	 * @param dataValue : The feature values and target value of the Data Point
	 * @return the standard error of the predicted value and the target
	 * value of the given Data Point.
	 */
	public Double calculateSEOfPredictedValue(double dataValue[]){
		
		Node node = this.rootNode;
		
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
				Double actualTargetValue = dataValue[Constant.HOUSING_DATA_TARGET_VALUE_INDEX];
				Double predictedTargetValue = node.getLabelValue();
				System.out.printf("%s\t%f %s\t%f\n","Actual Value",actualTargetValue,"Predicted Value",predictedTargetValue);
				return Math.pow(actualTargetValue-predictedTargetValue,2);
			}
		}
		
		
	}
	
	public static void main(String args[]){
		
		// Basic Regression Tree
		BasicRegressionTree basicRegressionTree = new BasicRegressionTree();
		basicRegressionTree.formRegressionTree();
		//System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
		basicRegressionTree.printRegressionTree();
		basicRegressionTree.evaluateTestDataSet();
		
	}
	
}
